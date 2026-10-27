"""Research Agent: cross-paper answers from the vector index.

Distinguishing tool: `corpus_search` (vector top-k across all papers through the LCEL retrieval chain).
Shared MCP tool: `tavily_search` (Tavily Remote MCP, the same tool Summary uses), here for current
developments on a topic. It is unlocked by observable state, not by the model's own choice:
  - the planner marked the task `needs_live` (current events, or a topic outside the corpus), or
  - corpus support is low: the best corpus_search similarity so far is below RAG_MIN_SCORE.
Both cases are traced as `tool_gate` events. If the web search fails, Research still returns its corpus
findings with status `partial` (error_kind live_search_unavailable).
"""

from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.agents.prompts import RESEARCH_ROLE
from app.agents.worker import run_tool_loop
from app.context import RequestContext
from app.mcp_tavily import shared_search_tool
from app.rag import AUGMENT_SYSTEM, MAX_K, MIN_K, cited_ids, make_corpus_search_chain

AGENT = "research"


class CorpusSearchArgs(BaseModel):
    query: str = Field(..., min_length=2, max_length=300, description="Focused search query in natural language")
    k: int | None = Field(None, ge=MIN_K, le=MAX_K, description="Number of passages to return (default 6)")


def corpus_search_tool(ctx: RequestContext, agent: str, best: dict | None = None) -> StructuredTool:
    """`best`, if given, records the highest similarity seen (the low-corpus-support signal)."""
    chain = make_corpus_search_chain(ctx.retriever, ctx, agent)

    async def corpus_search(query: str, k: int | None = None):
        out = await chain.ainvoke({"query": query, "k": k or ctx.k})
        if best is not None:
            top = max((r.score for r in out["refs"]), default=0.0)
            best["score"] = max(best.get("score") or 0.0, top)
        return out["evidence"], [r.cite_id for r in out["refs"]]

    return StructuredTool.from_function(
        coroutine=corpus_search,
        name="corpus_search",
        description="Vector search over the full text of the indexed arXiv papers. Returns the top-k most "
        "similar passages as <evidence> items with citation ids (C#), paper ids and section names.",
        args_schema=CorpusSearchArgs,
        response_format="content_and_artifact",
    )


async def run_research(ctx: RequestContext, task: dict, upstream: str) -> dict:
    system = AUGMENT_SYSTEM.format(
        worker_name="Research Agent",
        role_paragraph=RESEARCH_ROLE,
        retrieved_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )
    needs_live = bool(task.get("needs_live"))
    user = (f"Question: {ctx.query}\nSupervisor task: {task['instruction']}\n"
            f"Live web search: {'available (task needs current information)' if needs_live else 'not available unless corpus support is low'}\n"
            f"Upstream results: {upstream or 'none'}")
    best: dict = {"score": None}
    corpus = corpus_search_tool(ctx, AGENT, best)
    web = shared_search_tool(ctx, AGENT, purpose="find current developments on the topic")
    gate = research_gate(ctx, needs_live, best, corpus, web)
    loop = await run_tool_loop(
        ctx,
        agent=AGENT,
        system=system,
        user=user,
        tools=[corpus, web],
        max_rounds=ctx.settings.max_tool_rounds,
        gate=gate,
    )
    used = [c for c in cited_ids(loop.text) if ctx.registry.get(c)]
    unknown = [c for c in cited_ids(loop.text) if not ctx.registry.get(c)]
    corpus_errors = [e for e in loop.tool_errors if not e.startswith("tavily_search")]
    web_errors = [e for e in loop.tool_errors if e.startswith("tavily_search")]
    if not used and corpus_errors:
        status, error_kind = "failed", "tool_error"
    elif not used and web_errors and best["score"] is None:
        status, error_kind = "failed", "live_search_unavailable"
    elif loop.tool_errors or unknown or not loop.text.strip():
        status = "partial"
        error_kind = ("tool_error" if corpus_errors else "live_search_unavailable" if web_errors
                      else "citation_or_empty")
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
        "best_corpus_score": best["score"],
        "web_enabled": gate.enabled,
    }


class research_gate:
    """Per-round tool availability for Research (see the module docstring); traced as tool_gate events."""

    def __init__(self, ctx: RequestContext, needs_live: bool, best: dict, corpus, web):
        self.ctx, self.best, self.corpus, self.web = ctx, best, corpus, web
        self.enabled = needs_live
        ctx.emitter.emit("tool_gate", agent=AGENT, tool="tavily_search", enabled=needs_live,
                         reason="planner marked the task needs_live" if needs_live
                         else "corpus first; web unlocks if corpus support is low")

    def __call__(self, out) -> tuple[list, str | None]:
        note = None
        score, floor = self.best["score"], self.ctx.settings.rag_min_score
        if not self.enabled and score is not None and score < floor:
            self.enabled = True
            reason = f"low corpus support: best similarity {score:.2f} < RAG_MIN_SCORE {floor:.2f}"
            self.ctx.emitter.emit("tool_gate", agent=AGENT, tool="tavily_search", enabled=True, reason=reason,
                                  best_corpus_score=score)
            note = (f"Corpus support is low ({reason}). tavily_search is now available: use it once or twice "
                    "for web evidence and cite [W#] markers; say plainly that the indexed papers do not cover "
                    "this. Do not cite corpus passages that are off-topic.")
        return ([self.corpus, self.web] if self.enabled else [self.corpus]), note
