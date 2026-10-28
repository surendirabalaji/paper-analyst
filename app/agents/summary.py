"""Summary Agent: faithful, section-by-section summaries of specific papers, and a paper's current status.

Distinguishing tools (one paper at a time, read in document order, not a corpus-wide similarity search):
  get_paper_outline(paper)              metadata from SQLite + the section list from the index, abstract as C#
  get_paper_section(paper, section)     every chunk of one section, in order, as C# evidence (full text)
  search_within_paper(paper, query, k)  vector top-k restricted to one paper
Shared MCP tool: `tavily_search` (the same Tavily Remote MCP tool Research uses), here only for the
paper's *current status*: venue or acceptance, newer versions, follow-up work. It is available when the
planner marked the task `needs_live` or the question asks about status (see `status_cue`).

`paper` accepts an arXiv id (with or without version) or a distinctive title fragment; papers outside the
corpus resolve to a clear "not in corpus" result, and the task ends `partial`.
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
from datetime import datetime, timezone

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.agents.prompts import SUMMARY_ROLE
from app.agents.worker import run_tool_loop
from app.context import RequestContext
from app.llm.policy import approx_tokens
from app.mcp_tavily import shared_search_tool
from app.rag import AUGMENT_SYSTEM, MAX_K, MIN_K, cited_ids, format_evidence

AGENT = "summary"
SECTION_TOKEN_CAP = 3500  # one get_paper_section call returns at most this much text (in reading order)
ARXIV_RE = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")
# Only words about a paper's publication status; generic recency words ("current", "now", a year) are left
# to the planner's needs_live flag, so plain summary tasks do not reach the web (P5-N2).
_STATUS = re.compile(
    r"\b(status|venue|accepted|acceptance|published (in|at)|conference|journal|peer[- ]review(ed)?|"
    r"follow[- ]?ups?|follow[- ]up work|superseded|newer version|latest version|cited by)\b",
    re.I,
)


def status_cue(text: str) -> bool:
    """Observable cue that a Summary task asks about a paper's current status (enables tavily_search)."""
    return bool(_STATUS.search(text or ""))


class PaperNotFound(Exception):
    kind = "paper_not_found"


def _connect(path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def resolve_paper(db_path, paper: str) -> dict:
    """arXiv id or title fragment -> the papers row. Raises PaperNotFound with candidates when ambiguous."""
    q = (paper or "").strip().removeprefix("arXiv:").removeprefix("arxiv:").strip()
    if len(q) < 3:
        raise PaperNotFound(f"paper reference {paper!r} is too short")
    with _connect(db_path) as con:
        m = ARXIV_RE.match(q)
        if m:
            row = con.execute("SELECT * FROM papers WHERE arxiv_id = ?", (m.group(1),)).fetchone()
            if row:
                return dict(row)
            raise PaperNotFound(f"arXiv:{q} is not in the indexed corpus")
        like = "%" + q.replace("%", "").replace("_", "\\_") + "%"
        rows = con.execute(
            "SELECT * FROM papers WHERE title LIKE ? ESCAPE '\\' ORDER BY length(title) LIMIT 6", (like,)
        ).fetchall()
    if not rows:
        raise PaperNotFound(f"no indexed paper title contains {q!r}; it is not in the corpus")
    # A fragment wins only if it is the whole title or the short name before the colon ("Self-RAG" in
    # "Self-RAG: Learning to ..."), so "Agent" does not silently pick "AgentBench: ...".
    ql = q.lower().rstrip(":").strip()
    exact = [r for r in rows if r["title"].lower() == ql or r["title"].lower().split(":")[0].strip() == ql]
    if len(rows) == 1 or len(exact) == 1:
        return dict((exact or rows)[0])
    names = "; ".join(f"{r['arxiv_id']} {r['title'][:70]}" for r in rows)
    raise PaperNotFound(f"{q!r} matches several papers, use an arXiv id: {names}")


def _authors(db_path, arxiv_id: str) -> list[str]:
    with _connect(db_path) as con:
        rows = con.execute(
            "SELECT a.name FROM paper_authors pa JOIN authors a USING (author_id) WHERE pa.arxiv_id = ? "
            "ORDER BY pa.position", (arxiv_id,),
        ).fetchall()
    return [r["name"] for r in rows]


class PaperArgs(BaseModel):
    paper: str = Field(..., min_length=3, max_length=200, description="arXiv id (e.g. 2310.11511) or a distinctive title fragment")


class SectionArgs(PaperArgs):
    section: str = Field(..., min_length=1, max_length=120, description="section_id or section title from get_paper_outline")


class WithinArgs(PaperArgs):
    query: str = Field(..., min_length=2, max_length=300, description="What to look for inside this paper")
    k: int | None = Field(None, ge=MIN_K, le=MAX_K, description="Number of passages (default 6)")


def summary_tools(ctx: RequestContext) -> list[StructuredTool]:
    db = ctx.settings.sqlite_path
    retriever = ctx.retriever

    def emit_retrieval(tool: str, query: str, refs, ms: int) -> None:
        ctx.emitter.emit("retrieval", agent=AGENT, tool=tool, query=query, k=len(refs), retrieval_ms=ms,
                         chunks=[r.public() for r in refs])

    async def _paper(paper: str) -> dict:
        return await asyncio.to_thread(resolve_paper, db, paper)

    async def get_paper_outline(paper: str):
        p = await _paper(paper)
        t0 = datetime.now(timezone.utc)
        chunks = await retriever.paper_chunks(p["doc_id"])
        ms = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)
        sections: dict[str, dict] = {}
        for c in chunks:
            sid = c.metadata.get("section_id", "")
            s = sections.setdefault(sid, {"section_id": sid, "title": c.metadata.get("section_path", sid),
                                          "chunks": 0, "tokens": 0})
            s["chunks"] += 1
            s["tokens"] += int(c.metadata.get("n_tokens", 0))
        abstract = [ctx.registry.register_chunk(c, scope="paper") for c in chunks if c.metadata.get("section_id") == "abstract"]
        ctx.retrieval_ms_sum += ms
        emit_retrieval("get_paper_outline", p["arxiv_id"], abstract, ms)
        authors = await asyncio.to_thread(_authors, db, p["arxiv_id"])
        lines = [
            f"Paper arXiv:{p['doc_id']}: {p['title']}",
            f"Authors: {', '.join(authors[:8])}{' et al.' if len(authors) > 8 else ''}",
            f"Submitted {p['published']} (indexed version {p['updated']}); category {p['primary_category']}"
            + (f"; journal_ref: {p['journal_ref']}" if p.get("journal_ref") else "")
            + (f"; comment: {p['comment'][:160]}" if p.get("comment") else ""),
            "Sections (section_id | title | chunks | ~tokens):",
            *[f"- {s['section_id']} | {s['title']} | {s['chunks']} | {s['tokens']}" for s in sections.values()],
            "Abstract:",
            format_evidence(abstract),
        ]
        return "\n".join(lines), [r.cite_id for r in abstract]

    async def get_paper_section(paper: str, section: str):
        p = await _paper(paper)
        t0 = datetime.now(timezone.utc)
        chunks = await retriever.paper_chunks(p["doc_id"], section.strip())
        if not chunks:  # accept a title ("Method", "3 Experiments") as well as a section_id
            want = section.strip().lower()
            every = await retriever.paper_chunks(p["doc_id"])
            ids = [c.metadata.get("section_id") for c in every
                   if want in str(c.metadata.get("section_path", "")).lower()
                   or want == str(c.metadata.get("section_title", "")).lower()]
            first = ids[0] if ids else None
            chunks = [c for c in every if c.metadata.get("section_id") == first] if first else []
        if not chunks:
            raise PaperNotFound(f"section {section!r} not found in arXiv:{p['doc_id']}; call get_paper_outline")
        kept, used = [], 0
        for c in chunks:
            n = approx_tokens(c.text)
            if kept and used + n > SECTION_TOKEN_CAP:
                break
            kept.append(c)
            used += n
        ms = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)
        ctx.retrieval_ms_sum += ms
        refs = [ctx.registry.register_chunk(c, scope="paper") for c in kept]
        emit_retrieval("get_paper_section", f"{p['arxiv_id']} / {section}", refs, ms)
        more = f"\n(section truncated: {len(chunks) - len(kept)} more chunks not shown)" if len(kept) < len(chunks) else ""
        return format_evidence(refs) + more, [r.cite_id for r in refs]

    async def search_within_paper(paper: str, query: str, k: int | None = None):
        p = await _paper(paper)
        t0 = datetime.now(timezone.utc)
        found = await retriever.search(query, k or ctx.k, doc_id=p["doc_id"])
        ms = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)
        ctx.retrieval_ms_sum += ms
        refs = [ctx.registry.register_chunk(c, scope="paper") for c in found]
        emit_retrieval("search_within_paper", f"{p['arxiv_id']}: {query}", refs, ms)
        return format_evidence(refs), [r.cite_id for r in refs]

    return [
        StructuredTool.from_function(
            coroutine=get_paper_outline, name="get_paper_outline", args_schema=PaperArgs,
            response_format="content_and_artifact",
            description="Metadata (title, authors, dates, venue comment) and the ordered section list of ONE "
            "indexed paper, plus its abstract as <evidence> (C#). Call this first.",
        ),
        StructuredTool.from_function(
            coroutine=get_paper_section, name="get_paper_section", args_schema=SectionArgs,
            response_format="content_and_artifact", metadata={"reads_full_text": True},
            description="Full text of one section of one paper, in reading order, as <evidence> items (C#). "
            f"Capped at about {SECTION_TOKEN_CAP} tokens per call.",
        ),
        StructuredTool.from_function(
            coroutine=search_within_paper, name="search_within_paper", args_schema=WithinArgs,
            response_format="content_and_artifact",
            description="Vector search restricted to ONE paper; returns its most relevant passages (C#).",
        ),
    ]


def handoff_line(papers: list[dict] | None) -> str:
    """Upstream paper ids (from the Data agent) as one line of the task input."""
    if not papers:
        return "none"
    return "; ".join(f"arXiv:{p['arxiv_id']} \"{p.get('title') or ''}\""
                     + (f" (published {p['published']})" if p.get("published") else "")
                     + (f" [{p['cite_id']}]" if p.get("cite_id") else "") for p in papers)


async def run_summary(ctx: RequestContext, task: dict, upstream: str) -> dict:
    live = bool(task.get("needs_live")) or status_cue(task["instruction"])
    tools = summary_tools(ctx)
    if live:
        tools.append(shared_search_tool(ctx, AGENT, purpose="check the current status of a specific paper"))
    ctx.emitter.emit("tool_gate", agent=AGENT, tool="tavily_search", enabled=live,
                     reason=("planner marked the task needs_live" if task.get("needs_live")
                             else "task asks about current status" if live else "no status or live cue in the task"))
    system = AUGMENT_SYSTEM.format(
        worker_name="Summary Agent",
        role_paragraph=SUMMARY_ROLE,
        retrieved_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )
    user = (f"Question: {ctx.query}\nSupervisor task: {task['instruction']}\n"
            f"Web status check available: {'yes' if live else 'no'}\n"
            f"Papers handed off: {handoff_line(task.get('upstream_papers'))}\nUpstream results: {upstream or 'none'}")
    evidence_only = bool(task.get("evidence_only"))
    loop = await run_tool_loop(ctx, agent=AGENT, system=system, user=user, tools=tools,
                               max_rounds=ctx.settings.max_tool_rounds, seeds=task.get("seeds"),
                               evidence_only=evidence_only)
    # Evidence-only (one-task plans): the sections and abstracts it read go to synthesis as they are.
    used = [c for c in (loop.evidence_ids if evidence_only else cited_ids(loop.text)) if ctx.registry.get(c)]
    unknown = [] if evidence_only else [c for c in cited_ids(loop.text) if not ctx.registry.get(c)]
    paper_errors = [e for e in loop.tool_errors if e.endswith("paper_not_found")]
    if not used and loop.tool_errors:
        status, error_kind = ("partial" if paper_errors else "failed"), (
            "paper_not_found" if paper_errors else "tool_error")
    elif loop.tool_errors or unknown or (not used if evidence_only else not loop.text.strip()):
        web_only = loop.tool_errors and all(e.startswith("tavily_search") for e in loop.tool_errors)
        status = "partial"
        error_kind = ("live_search_unavailable" if web_only else "paper_not_found" if paper_errors
                      else "tool_error" if loop.tool_errors else "citation_or_empty")
    else:
        status, error_kind = "done", None
    return {
        "status": status,
        "error_kind": error_kind,
        "findings_md": loop.text,
        "evidence_ids": used,
        "unknown_markers": unknown,
        "rounds": loop.rounds,
        "tool_calls": loop.tool_calls,
        "tool_errors": loop.tool_errors,
        "evidence_only": evidence_only,
        "seeded_calls": loop.seeded_calls,
        "long_source_tokens": loop.long_source_tokens,
    }
