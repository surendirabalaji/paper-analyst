"""Data Agent: exact metadata questions over the corpus database (`data/papers.db`, SQLite).

Distinguishing tools (no other worker has them; Data has no web or vector-search tool):
  get_schema()      DDL of the five corpus tables plus value hints (years, categories, topics)
  run_sql(query)    one read-only SELECT through app.sqlguard (mode=ro, authorizer, LIMIT 50, op cap)

Every successful query becomes D# evidence in the request registry: one D# per paper row (linked to its
arXiv abstract page) or one D# for an aggregate result. An empty result is a valid answer ("no such papers")
and gets its own D#. Each D# keeps its SQL, which the trace and the D# viewer show.

Invalid SQL (unknown column, syntax) or a refused statement gets one repair round: the model sees the
error and may send one corrected query. If that also fails, run_sql is closed for the task and the task
ends `failed` (error_kind invalid_sql). The repair call is routed by the `repair_round` feature (policy.py).

Hand-off: the result lists the paper ids it found (`paper_ids`, cited rows first), which the supervisor
passes to dependent tasks (for example Summary) as `upstream_papers`.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.agents.prompts import DATA_ROLE
from app.agents.worker import run_tool_loop
from app.context import RequestContext
from app.rag import AUGMENT_SYSTEM, cited_ids, format_evidence
from app.sqlguard import MAX_ROWS, MAX_SQL_CHARS, SqlError, run_readonly, schema_text

AGENT = "data"
MAX_SQL_FAILURES = 2  # the first failure gets one repair round; the second ends the task
MAX_HANDOFF_PAPERS = 10


class SqlUnavailable(Exception):
    kind = "repair_budget_spent"


class SchemaArgs(BaseModel):
    pass


class SqlArgs(BaseModel):
    query: str = Field(..., min_length=6, max_length=MAX_SQL_CHARS,
                       description="One read-only SQLite SELECT (or WITH ... SELECT) statement, no comments")


def data_tools(ctx: RequestContext, state: dict) -> list[StructuredTool]:
    """`state` collects this task's outcome: successful refs (in row order), failures, last error."""
    db = ctx.settings.sqlite_path

    async def get_schema() -> str:
        return await asyncio.to_thread(schema_text, db)

    async def run_sql(query: str):
        if state["failures"] >= MAX_SQL_FAILURES:
            raise SqlUnavailable("run_sql is closed for this task: the query failed and the one repair round "
                                 "was used. Report the failure; do not guess the answer.")
        try:
            res = await asyncio.to_thread(run_readonly, db, query)
        except SqlError as e:
            state["failures"] += 1
            state["last_error"] = e.kind
            left = MAX_SQL_FAILURES - state["failures"]
            ctx.emitter.emit("sql_result", agent=AGENT, status="error", error_kind=e.kind, sql=query[:MAX_SQL_CHARS],
                             message_safe=str(e)[:300], repair_rounds_left=left)
            hint = (" You have one repair attempt: fix the query using get_schema." if left
                    else " No repair attempts left.")
            raise SqlError(e.kind, f"{e}.{hint}") from None
        ctx.retrieval_ms_sum += res.elapsed_ms
        refs = ctx.registry.register_sql(sql=res.sql, columns=res.columns, rows=res.rows, agent=AGENT)
        state["refs"].extend(r for r in refs if r.cite_id not in {x.cite_id for x in state["refs"]})
        state["ok"] += 1
        state["rows"] += len(res.rows)
        ctx.emitter.emit(
            "sql_result", agent=AGENT, status="ok", sql=res.sql, executed_sql=res.executed_sql,
            columns=res.columns, rows=res.rows, n_rows=len(res.rows), truncated=res.truncated,
            elapsed_ms=res.elapsed_ms, cite_ids=[r.cite_id for r in refs],
            refs=[{"cite_id": r.cite_id, "type": r.type, "doc_id": r.doc_id, "title": r.title, "href": r.href,
                   "rows": r.rows} for r in refs],
        )
        note = (f"{len(res.rows)} row(s)" + (f"; more rows exist, only the first {MAX_ROWS} are shown" if res.truncated else "")
                + ("; an empty result means the corpus has no such records" if not res.rows else ""))
        return f"SQL: {res.sql}\nResult: {note}\n{format_evidence(refs)}", [r.cite_id for r in refs]

    return [
        StructuredTool.from_function(
            coroutine=get_schema, name="get_schema", args_schema=SchemaArgs,
            description="Schema (CREATE TABLE statements) of the corpus metadata database plus value hints: "
            "years, category and topic values, how repos are recorded. Call this before writing SQL.",
        ),
        StructuredTool.from_function(
            coroutine=run_sql, name="run_sql", args_schema=SqlArgs, response_format="content_and_artifact",
            description=f"Run ONE read-only SQLite SELECT over papers, authors, paper_authors, paper_categories "
            f"and repos. At most {MAX_ROWS} rows are returned; results come back as <evidence> items with D# "
            "markers. Include arxiv_id and title columns when listing papers.",
        ),
    ]


def handoff_papers(refs: list, cited: list[str]) -> list[dict]:
    """Paper rows for dependent tasks: cited rows first (in citation order), then the rest in row order."""
    rows = [r for r in refs if r.type == "sql_row"]
    order = {c: i for i, c in enumerate(cited)}
    rows.sort(key=lambda r: (0, order[r.cite_id]) if r.cite_id in order else (1, 0))
    seen, out = set(), []
    for r in rows:
        rec = dict(zip([c.lower() for c in r.columns or []], (r.rows or [[]])[0]))
        aid = str(rec.get("arxiv_id") or "")
        if not aid or aid in seen:
            continue
        seen.add(aid)
        out.append({"arxiv_id": aid, "title": r.title, "published": rec.get("published"), "cite_id": r.cite_id})
    return out[:MAX_HANDOFF_PAPERS]


async def run_data(ctx: RequestContext, task: dict, upstream: str) -> dict:
    state = {"refs": [], "ok": 0, "rows": 0, "failures": 0, "last_error": None}
    system = AUGMENT_SYSTEM.format(
        worker_name="Data Agent",
        role_paragraph=DATA_ROLE,
        retrieved_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )
    user = f"Question: {ctx.query}\nSupervisor task: {task['instruction']}\nUpstream results: {upstream or 'none'}"
    loop = await run_tool_loop(ctx, agent=AGENT, system=system, user=user, tools=data_tools(ctx, state),
                               max_rounds=ctx.settings.max_tool_rounds)
    mine = {r.cite_id for r in state["refs"]}
    cited = cited_ids(loop.text)
    used = [c for c in cited if c in mine]
    unknown = [c for c in cited if not ctx.registry.get(c)]
    if not state["ok"]:
        status = "failed"
        error_kind = "invalid_sql" if state["failures"] else "tool_error" if loop.tool_errors else "no_query"
    elif not used or unknown or not loop.text.strip():
        # The query ran, so its rows are real evidence even if the write-up did not cite them.
        status, error_kind = "partial", "citation_or_empty"
        used = used or [r.cite_id for r in state["refs"]][:MAX_HANDOFF_PAPERS]
    else:
        status, error_kind = "done", None
    return {
        "status": status,
        "error_kind": error_kind,
        "findings_md": loop.text,
        "evidence_ids": used,
        "unknown_markers": unknown,
        "paper_ids": handoff_papers(state["refs"], cited) if state["ok"] else [],
        "sql_ok": state["ok"],
        "sql_rows": state["rows"],  # rows returned by successful queries (0 = "no such records")
        "sql_failures": state["failures"],
        "rounds": loop.rounds,
        "tool_calls": loop.tool_calls,
        "tool_errors": loop.tool_errors,
    }
