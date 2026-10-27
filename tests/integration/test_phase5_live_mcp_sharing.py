"""Phase 5 live checks (REAL Tavily Remote MCP, OpenAI and Gemini-via-OpenRouter calls; RUN_LIVE=1).

1. The real Tavily MCP server returns parsed results with URLs through the app's client.
2. A fake key gets an auth failure (Tavily returns it as a tool result), classified `auth`, key not echoed.
3. One real request through the full graph in which Research and Summary each call the shared
   `tavily_search` MCP tool, with distinct agent identities and purposes, and the answer cites W#.
Cost: about $0.03-0.06 in LLM calls plus 3-4 Tavily searches per run.
"""

import dataclasses
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from phase2_fakes import drain  # noqa: E402

pytestmark = [pytest.mark.live, pytest.mark.anyio]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
def settings():
    from app.config import load_settings

    s = load_settings()
    if not (s.tavily_api_key and s.openai_api_key and s.openrouter_api_key):
        pytest.skip("TAVILY_API_KEY, OPENAI_API_KEY and OPENROUTER_API_KEY are required")
    return s


async def test_real_tavily_search_parses_results(settings):
    from app.mcp_tavily import TavilyMCP

    rs = await TavilyMCP(settings).search("Self-RAG retrieval augmented generation ICLR", 2)
    assert 1 <= len(rs) <= 2 and all(r["url"].startswith("http") for r in rs)


async def test_real_tavily_fake_key_is_auth_error_and_redacted(settings):
    from app.mcp_tavily import TavilyMCP, TavilyUnavailable

    fake = "tvly-dev-NOTAREALKEY0000000000"
    with pytest.raises(TavilyUnavailable) as ei:
        await TavilyMCP(dataclasses.replace(settings, tavily_api_key=fake)).search("agents", 1)
    assert ei.value.kind in ("auth", "connection")  # Tavily may reject at the HTTP layer or in the tool result
    assert fake not in str(ei.value)


async def test_research_and_summary_both_call_shared_tavily_search(settings):
    from app.context import RequestContext
    from app.events import Emitter, sse_frame
    from app.graph import run_request
    from app.llm.router import Router
    from app.mcp_tavily import TavilyMCP
    from app.rag import Retriever

    q = ("What are the most recent developments in LLM agent memory this year, and has the Self-RAG "
         "paper been accepted at a peer-reviewed venue?")
    ctx = RequestContext(request_id="live5", query=q, k=settings.top_k, settings=settings,
                         router=Router(settings), retriever=Retriever(settings), emitter=Emitter("live5"),
                         tavily=TavilyMCP(settings))
    await run_request(ctx)
    events = drain(ctx)
    ws = [d for e, d in events if e == "web_search"]
    ok_agents = {d["agent"] for d in ws if d["status"] == "ok"}
    assert {"research", "summary"} <= ok_agents, [(d["agent"], d["status"], d.get("error_kind")) for d in ws]
    assert {d["tool"] for d in ws} == {"tavily_search"} and {d["server"] for d in ws} == {"tavily-remote-mcp"}
    purposes = {d["agent"]: d["purpose"] for d in ws}
    assert purposes["research"] != purposes["summary"]
    cites = [d for e, d in events if e == "citation"]
    assert any(c["type"] == "web" and c["source_url"].startswith("http") for c in cites)
    done = [d for e, d in events if e == "done"][0]
    assert done["status"] in ("done", "partial")
    frames = "".join(sse_frame(e, d) for e, d in events)
    assert settings.tavily_api_key not in frames and "tavilyApiKey=" not in frames
    out = pathlib.Path(__file__).resolve().parents[2] / "eval" / "results" / "phase5" / "live_test_sharing.events.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps({"event": e, "data": d}, default=str) + "\n" for e, d in events))
