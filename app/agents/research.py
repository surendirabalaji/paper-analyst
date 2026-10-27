"""Research Agent: cross-paper answers from the vector index.

Distinguishing tool: `corpus_search` (vector top-k across all papers through the LCEL retrieval chain).
Phase 5 adds the shared Tavily `tavily_search` MCP tool alongside it.
"""

from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.agents.prompts import RESEARCH_ROLE
from app.agents.worker import run_tool_loop
from app.context import RequestContext
from app.rag import AUGMENT_SYSTEM, MAX_K, MIN_K, cited_ids, make_corpus_search_chain

AGENT = "research"


class CorpusSearchArgs(BaseModel):
    query: str = Field(..., min_length=2, max_length=300, description="Focused search query in natural language")
    k: int | None = Field(None, ge=MIN_K, le=MAX_K, description="Number of passages to return (default 6)")


def corpus_search_tool(ctx: RequestContext, agent: str) -> StructuredTool:
    chain = make_corpus_search_chain(ctx.retriever, ctx, agent)

    async def corpus_search(query: str, k: int | None = None):
        out = await chain.ainvoke({"query": query, "k": k or ctx.k})
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
    user = f"Question: {ctx.query}\nSupervisor task: {task['instruction']}\nUpstream results: {upstream or 'none'}"
    loop = await run_tool_loop(
        ctx,
        agent=AGENT,
        system=system,
        user=user,
        tools=[corpus_search_tool(ctx, AGENT)],
        max_rounds=ctx.settings.max_tool_rounds,
    )
    used = [c for c in cited_ids(loop.text) if ctx.registry.get(c)]
    unknown = [c for c in cited_ids(loop.text) if not ctx.registry.get(c)]
    if not ctx.registry.by_cite and loop.tool_errors:
        status, error_kind = "failed", "tool_error"
    elif loop.tool_errors or unknown or not loop.text.strip():
        status, error_kind = "partial", ("tool_error" if loop.tool_errors else "citation_or_empty")
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
    }
