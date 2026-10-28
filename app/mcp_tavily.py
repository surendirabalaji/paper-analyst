"""Tavily Remote MCP client and the shared `tavily_search` tool (live retrieval + MCP tool sharing).

Transport: hosted Tavily MCP over Streamable HTTP via langchain-mcp-adapters. The API key is sent only as
the `tavilyApiKey` query parameter of the server URL (never in a request body). The URL is held in this
process only: it is never logged, traced or returned, and every error string passes through `redact()`.

One `TavilyMCP` per process loads the server's tool list once; each tool call opens its own short MCP
session, so concurrent requests share nothing mutable. Each worker gets its own wrapper around the
*same* MCP tool (`shared_search_tool`), so the trace shows which agent called it and why.

Tavily reports auth and quota failures as an ordinary tool result (`{"error": ..., "status": 401}`), not
as an exception (Phase 0, P0-D1). A call therefore counts as successful only when the result parses into
at least one item with an http(s) URL (P0-I4). Everything else raises `TavilyUnavailable` with a kind the
worker loop records as a recoverable tool error; workers then continue from corpus evidence.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from app.config import Settings
from app.events import redact, safe_error, utc_now

SERVER = "tavily-remote-mcp"
TOOL = "tavily_search"
MAX_CALLS_PER_TASK = 2
SNIPPET_CHARS = 1200  # per result, kept in the evidence registry and the prompt


class TavilyUnavailable(Exception):
    """A tavily_search call produced no usable results. `kind`: auth | quota | rate_limit | connection |
    timeout | empty | upstream_error | disabled | budget."""

    def __init__(self, kind: str, message: str):
        super().__init__(redact(message))
        self.kind = kind


def classify_status(status: int) -> str:
    if status in (401, 403):
        return "auth"
    if status == 429:
        return "rate_limit"
    if status in (402, 432, 433):  # Tavily: plan / pay-as-you-go credit limits
        return "quota"
    return "upstream_error"


def _blocks_to_payloads(raw: Any) -> list[Any]:
    """Normalise what the adapter returns (content blocks, a JSON string, a dict, or (content, artifact))."""
    if isinstance(raw, tuple) and len(raw) == 2:  # content_and_artifact
        content, artifact = raw
        if isinstance(artifact, dict) and isinstance(artifact.get("structured_content"), dict):
            return [artifact["structured_content"]]
        raw = content
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, str):
        raw = [{"type": "text", "text": raw}]
    out = []
    for b in raw or []:
        text = b.get("text") if isinstance(b, dict) else (b if isinstance(b, str) else None)
        if text is None:
            continue
        try:
            out.append(json.loads(text))
        except (json.JSONDecodeError, TypeError):
            out.append(text)
    return out


def parse_search_result(raw: Any) -> list[dict]:
    """Parse a tavily_search result into [{url, title, content, score, published_date}].

    Raises TavilyUnavailable for error payloads, non-JSON text and results without URLs.
    """
    payloads = _blocks_to_payloads(raw)
    if not payloads:
        raise TavilyUnavailable("empty", "tavily_search returned no content")
    results: list[dict] = []
    for p in payloads:
        if not isinstance(p, dict):
            raise TavilyUnavailable("upstream_error", f"unparseable tavily_search result: {str(p)[:160]}")
        status = p.get("status")
        if p.get("error") or (isinstance(status, int) and status >= 400):
            kind = classify_status(status) if isinstance(status, int) else "upstream_error"
            raise TavilyUnavailable(kind, f"Tavily error status={status}: {str(p.get('error'))[:160]}")
        for r in p.get("results") or []:
            url = (r or {}).get("url") or ""
            if not url.startswith(("http://", "https://")):
                continue
            results.append(
                {
                    "url": url,
                    "title": str(r.get("title") or "")[:200],
                    "content": str(r.get("content") or "")[:SNIPPET_CHARS],
                    "score": r.get("score") or 0.0,
                    "published_date": r.get("published_date"),
                }
            )
    if not results:
        raise TavilyUnavailable("empty", "tavily_search returned zero results with URLs")
    return results


def _detail(e: BaseException) -> str:
    """Redacted message without stacking 'TavilyUnavailable:' prefixes when re-raised."""
    return str(e) if isinstance(e, TavilyUnavailable) else safe_error(e)


def _exception_kind(e: BaseException) -> str:
    if isinstance(e, TavilyUnavailable):
        return e.kind
    if isinstance(e, (asyncio.TimeoutError, TimeoutError)):
        return "timeout"
    if isinstance(e, BaseExceptionGroup):
        return _exception_kind(e.exceptions[0]) if e.exceptions else "connection"
    status = getattr(getattr(e, "response", None), "status_code", None)
    if isinstance(status, int):
        return classify_status(status)
    return "connection"  # transport / session errors (DNS, refused, TLS, MCP handshake)


class TavilyMCP:
    """Process-wide handle on the Tavily Remote MCP server (tool list cached after the first load)."""

    def __init__(self, settings: Settings):
        self._key = settings.tavily_api_key
        self._base = settings.tavily_mcp_base_url
        self.timeout_s = settings.tavily_timeout_s
        self.max_results = settings.tavily_max_results
        self._tool: BaseTool | None = None
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self._key)

    def _connections(self) -> dict:
        # The only place the key is placed: the tavilyApiKey query parameter of the server URL.
        sep = "&" if "?" in self._base else "?"
        return {"tavily": {"transport": "streamable_http", "url": f"{self._base}{sep}tavilyApiKey={self._key}"}}

    async def search_tool(self) -> BaseTool:
        if not self.enabled:
            raise TavilyUnavailable("disabled", "TAVILY_API_KEY is not configured")
        if self._tool is not None:
            return self._tool
        async with self._lock:
            if self._tool is None:
                from langchain_mcp_adapters.client import MultiServerMCPClient

                try:
                    tools = await asyncio.wait_for(MultiServerMCPClient(self._connections()).get_tools(),
                                                   self.timeout_s)
                except Exception as e:
                    raise TavilyUnavailable(_exception_kind(e), "MCP list_tools failed: " + safe_error(e)) from None
                found = next((t for t in tools if t.name == TOOL), None)
                if found is None:
                    raise TavilyUnavailable("upstream_error", f"server does not expose {TOOL}")
                self._tool = found
        return self._tool

    async def search(self, query: str, max_results: int, time_range: str | None = None) -> list[dict]:
        """One bounded tavily_search call (one retry on a connection error). Raises TavilyUnavailable."""
        args: dict[str, Any] = {"query": query, "max_results": max(1, min(self.max_results, max_results))}
        if time_range:
            args["time_range"] = time_range
        last: BaseException | None = None
        for attempt in (1, 2):
            try:
                tool = await self.search_tool()
                raw = await asyncio.wait_for(tool.ainvoke(args), self.timeout_s)
                return parse_search_result(raw)
            except Exception as e:
                last = e
                if _exception_kind(e) != "connection" or attempt == 2:
                    break
                await asyncio.sleep(0.5)
        raise TavilyUnavailable(_exception_kind(last), _detail(last)) from None


class TavilySearchArgs(BaseModel):
    query: str = Field(..., min_length=2, max_length=300, description="Web search query")
    max_results: int = Field(4, ge=1, le=5, description="Number of web results (1-5)")
    time_range: str | None = Field(
        None, pattern="^(day|week|month|year)$", description="Optional recency filter: day, week, month or year"
    )


def shared_search_tool(ctx, agent: str, purpose: str) -> StructuredTool:
    """This agent's handle on the shared Tavily MCP `tavily_search` tool.

    Every handle calls the same MCP server tool; it registers results as W# evidence in the request
    registry and emits a `web_search` event with the calling agent's identity and purpose.
    """
    mcp: TavilyMCP | None = getattr(ctx, "tavily", None)
    calls = {"n": 0}

    async def tavily_search(query: str, max_results: int = 4, time_range: str | None = None):
        t0 = time.perf_counter()
        retrieved_at = utc_now()
        base = {"agent": agent, "tool": TOOL, "server": SERVER, "transport": "streamable_http",
                "purpose": purpose, "query": query[:200], "time_range": time_range}
        try:
            calls["n"] += 1
            if calls["n"] > MAX_CALLS_PER_TASK:
                raise TavilyUnavailable("budget", f"at most {MAX_CALLS_PER_TASK} web searches per task")
            if mcp is None:
                raise TavilyUnavailable("disabled", "Tavily MCP is not configured")
            results = await mcp.search(query, max_results, time_range)
        except Exception as e:
            kind = _exception_kind(e)
            ctx.emitter.emit("web_search", **base, status="error", error_kind=kind, message_safe=_detail(e),
                             recoverable=True, latency_ms=int((time.perf_counter() - t0) * 1000), results=[])
            raise TavilyUnavailable(kind, _detail(e)) from None
        refs = [
            ctx.registry.register_web(url=r["url"], title=r["title"], content=r["content"], score=r["score"],
                                      retrieved_at=retrieved_at, agent=agent)
            for r in results
        ]
        ctx.emitter.emit(
            "web_search", **base, status="ok", latency_ms=int((time.perf_counter() - t0) * 1000),
            results=[{"cite_id": r.cite_id, "url": r.source_url, "title": r.title, "domain": r.doc_id,
                      "score": r.score, "retrieved_at": r.retrieved_at, "snippet": r.text}
                     for r in refs],
        )
        from app.rag import format_evidence  # local import: rag imports nothing from here

        return format_evidence(refs), [r.cite_id for r in refs]

    return StructuredTool.from_function(
        coroutine=tavily_search,
        name=TOOL,
        description="Live web search through the Tavily Remote MCP server. Returns current web results as "
        "<evidence> items with citation ids (W#), URLs and the retrieval time. Use for information newer "
        "than, or outside, the indexed papers.",
        args_schema=TavilySearchArgs,
        response_format="content_and_artifact",
    )
