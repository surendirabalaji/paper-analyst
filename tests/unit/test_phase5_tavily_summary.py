"""Phase 5 offline tests: Tavily MCP result parsing, credential redaction, the shared tavily_search tool
(Research and Summary), Research's web gate, Summary tools over the real local index, worker-side routing,
and the P3-N3 abstention-citation rule.

Provider responses and the MCP server are faked (MOCKED); Summary tool tests read the real local Chroma
index and SQLite DB (no network, no embedding calls).
"""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.config import ROOT
from app.events import RedactingFilter, safe_error, sse_frame
from app.graph import abstention_only_markers, run_request
from app.llm import policy
from app.mcp_tavily import TavilyMCP, TavilyUnavailable, parse_search_result, shared_search_tool
from tests.phase2_fakes import FakeChat, FakeRetriever, Script, chunk, drain, make_ctx, make_settings, tool_call

FAKE_KEY = "tvly-dev-FAKEKEY0123456789abcdef"

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def ev(events, name):
    return [d for e, d in events if e == name]


def blocks(obj) -> list[dict]:
    return [{"type": "text", "text": json.dumps(obj)}]


GOOD = {"query": "q", "results": [
    {"url": "https://example.org/a", "title": "A", "content": "alpha " * 10, "score": 0.9},
    {"url": "https://example.org/b", "title": "B", "content": "beta", "score": 0.5},
]}


# ---------------------------------------------------------------- MCP result parsing (P0-D1, P0-I4)


def test_parse_content_blocks_success():
    rs = parse_search_result(blocks(GOOD))
    assert [r["url"] for r in rs] == ["https://example.org/a", "https://example.org/b"]
    assert rs[0]["title"] == "A" and rs[0]["score"] == 0.9


def test_parse_artifact_structured_content_and_plain_json_string():
    assert len(parse_search_result(("ignored", {"structured_content": GOOD}))) == 2
    assert len(parse_search_result(json.dumps(GOOD))) == 2


@pytest.mark.parametrize("status,kind", [(401, "auth"), (403, "auth"), (429, "rate_limit"), (432, "quota"),
                                         (433, "quota"), (500, "upstream_error")])
def test_error_payloads_are_never_results(status, kind):
    with pytest.raises(TavilyUnavailable) as ei:
        parse_search_result(blocks({"error": "nope", "status": status}))
    assert ei.value.kind == kind


@pytest.mark.parametrize("payload", [
    {"query": "q", "results": []},
    {"query": "q", "results": [{"url": "ftp://x", "title": "x"}, {"title": "no url"}]},
])
def test_zero_usable_results_is_empty(payload):
    with pytest.raises(TavilyUnavailable) as ei:
        parse_search_result(blocks(payload))
    assert ei.value.kind == "empty"


def test_non_json_text_is_not_success():
    with pytest.raises(TavilyUnavailable) as ei:
        parse_search_result([{"type": "text", "text": "Internal error at https://example.org"}])
    assert ei.value.kind == "upstream_error"


# ---------------------------------------------------------------- TavilyMCP.search bounds


class FakeMCPTool:
    name = "tavily_search"

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def ainvoke(self, args):
        self.calls.append(args)
        o = self.outcomes.pop(0)
        if isinstance(o, BaseException):
            raise o
        return o


def mcp_with(tool, **over) -> TavilyMCP:
    m = TavilyMCP(make_settings(tavily_api_key=FAKE_KEY, **over))
    m._tool = tool
    return m


async def test_search_success_caps_max_results():
    tool = FakeMCPTool(blocks(GOOD))
    rs = await mcp_with(tool, tavily_max_results=3).search("agents", 5)
    assert len(rs) == 2 and tool.calls == [{"query": "agents", "max_results": 3}]


async def test_connection_error_retried_once_then_reported():
    err = httpx.ConnectError(f"connect failed for https://mcp.tavily.com/mcp/?tavilyApiKey={FAKE_KEY}")
    tool = FakeMCPTool(err, err)
    with pytest.raises(TavilyUnavailable) as ei:
        await mcp_with(tool).search("agents", 3)
    assert ei.value.kind == "connection" and len(tool.calls) == 2
    assert FAKE_KEY not in str(ei.value)


async def test_quota_error_not_retried():
    tool = FakeMCPTool(blocks({"error": "credits exhausted", "status": 432}), blocks(GOOD))
    with pytest.raises(TavilyUnavailable) as ei:
        await mcp_with(tool).search("agents", 3)
    assert ei.value.kind == "quota" and len(tool.calls) == 1


async def test_timeout_is_bounded():
    class Slow(FakeMCPTool):
        async def ainvoke(self, args):
            await asyncio.sleep(5)

    with pytest.raises(TavilyUnavailable) as ei:
        await mcp_with(Slow(), tavily_timeout_s=0.05).search("agents", 3)
    assert ei.value.kind == "timeout"


async def test_disabled_without_key():
    with pytest.raises(TavilyUnavailable) as ei:
        await TavilyMCP(make_settings()).search("agents", 3)
    assert ei.value.kind == "disabled"


def test_key_only_in_query_parameter():
    m = TavilyMCP(make_settings(tavily_api_key=FAKE_KEY))
    conn = m._connections()["tavily"]
    assert conn["transport"] == "streamable_http"
    assert conn["url"] == f"https://mcp.tavily.com/mcp/?tavilyApiKey={FAKE_KEY}"
    assert set(conn) == {"transport", "url"}  # no headers, no body fields carrying the key


# ---------------------------------------------------------------- redaction


def test_safe_error_redacts_httpx_status_error_url():
    req = httpx.Request("POST", f"https://mcp.tavily.com/mcp/?tavilyApiKey={FAKE_KEY}")
    e = httpx.HTTPStatusError(f"Client error '401 Unauthorized' for url '{req.url}'", request=req,
                              response=httpx.Response(401, request=req))
    s = safe_error(e)
    assert FAKE_KEY not in s and "tavilyApiKey=REDACTED" in s


def test_safe_error_redacts_exception_groups_and_url_encoded_keys():
    g = ExceptionGroup("unhandled errors in a TaskGroup", [
        RuntimeError(f"GET https://mcp.tavily.com/mcp/?tavilyApiKey%3D{FAKE_KEY}&x=1 failed")])
    s = safe_error(g)
    assert "RuntimeError" in s and FAKE_KEY not in s


def test_logging_filter_redacts_args_and_tracebacks(caplog):
    lg = logging.getLogger("phase5.redaction.test")
    caplog.handler.addFilter(RedactingFilter())
    with caplog.at_level(logging.INFO, logger="phase5.redaction.test"):
        lg.info("calling %s", f"https://mcp.tavily.com/mcp/?tavilyApiKey={FAKE_KEY}")
        try:
            raise RuntimeError(f"boom tavilyApiKey={FAKE_KEY}")
        except RuntimeError:
            lg.exception("failed")
    assert FAKE_KEY not in caplog.text and "REDACTED" in caplog.text


async def test_real_mcp_client_connection_failure_never_leaks_key(caplog):
    """Real langchain-mcp-adapters client against a closed local port: error, logs and events stay clean."""
    import app.main  # noqa: F401  (applies the production logger levels and handler filters)

    m = TavilyMCP(make_settings(tavily_api_key=FAKE_KEY, tavily_mcp_base_url="http://127.0.0.1:9/mcp/",
                                tavily_timeout_s=5))
    ctx = make_ctx("q", Script())
    ctx.tavily = m
    tool = shared_search_tool(ctx, "research", "test")
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(TavilyUnavailable) as ei:
            await tool.coroutine(query="agents")
    events = drain(ctx)
    frames = "".join(sse_frame(e, d) for e, d in events)
    assert ei.value.kind == "connection"
    for text in (str(ei.value), frames, caplog.text):
        assert FAKE_KEY not in text
    ws = ev(events, "web_search")[0]
    assert ws["status"] == "error" and ws["recoverable"] is True and ws["agent"] == "research"


# ---------------------------------------------------------------- graph-level fakes for Phase 5


class FakeMCP:
    """Stands in for TavilyMCP: records (query) per call; outcomes are results lists or exceptions."""

    enabled = True

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[str] = []

    async def search(self, query, max_results, time_range=None):
        self.calls.append(query)
        o = self.outcomes.pop(0) if self.outcomes else parse_search_result(blocks(GOOD))
        if isinstance(o, BaseException):
            raise o
        return o


class Chat5(FakeChat):
    """Worker calls are scripted per agent (detected from the bound tool names); bound tools are recorded."""

    async def _agenerate(self, messages, stop=None, run_manager=None, **kw):
        names = kw.get("tool_names") or []
        s = self.script
        agent = "summary" if "get_paper_outline" in names else "research" if "corpus_search" in names else None
        if agent is None:
            return await super()._agenerate(messages, stop, run_manager, **kw)
        n = sum(1 for c in s.calls if c.get("agent") == agent)
        s.calls.append({"site": "worker", "agent": agent, "tools": names, "tool_choice": kw.get("tool_choice"),
                        "messages": list(messages)})
        seq = s.worker_by_agent.get(agent, [])
        item = seq[n] if n < len(seq) else AIMessage(content="no more script")
        item = await self._resolve(item, messages, kw)
        return ChatResult(generations=[ChatGeneration(message=item)])


def ctx5(query, script, retriever=None, mcp=None, **over):
    from tests.phase2_fakes import FakeRouter

    ctx = make_ctx(query, script, retriever, **over)

    class R(FakeRouter):
        def client(self, provider):
            return Chat5(script=script)

    ctx.router = R(ctx.settings, script)
    ctx.tavily = mcp
    return ctx


def worker_calls(script, agent):
    return [c for c in script.calls if c.get("agent") == agent]


def research_plan(needs_live=False):
    return {"plan_summary": "p", "tasks": [{"worker": "research", "instruction": "find it", "depends_on": [],
                                            "needs_live": needs_live}]}


# ---------------------------------------------------------------- Research web gate


async def test_research_without_needs_live_and_good_support_never_sees_web_tool():
    s = Script(planner=research_plan(False))
    s.worker_by_agent = {"research": [
        AIMessage(content="", tool_calls=[tool_call("corpus_search", {"query": "alpha"})]),
        AIMessage(content="- Alpha retrieves when uncertain [C1]."),
    ]}
    mcp = FakeMCP()
    ctx = ctx5("How does alpha work?", s, FakeRetriever(), mcp)  # default chunk score 0.8
    await run_request(ctx)
    events = drain(ctx)
    assert all(c["tools"] == ["corpus_search"] for c in worker_calls(s, "research"))
    assert mcp.calls == []
    gate = ev(events, "tool_gate")
    assert len(gate) == 1 and gate[0]["enabled"] is False


async def test_low_corpus_support_unlocks_web_in_next_round():
    low = FakeRetriever(lambda q: [chunk("2401.00001v1", "S1", 0, "Unrelated text.", score=0.21)])
    s = Script(planner=research_plan(False))
    s.worker_by_agent = {"research": [
        AIMessage(content="", tool_calls=[tool_call("corpus_search", {"query": "capital of australia"})]),
        AIMessage(content="", tool_calls=[tool_call("tavily_search", {"query": "capital of Australia"}, 2)]),
        AIMessage(content="- The indexed papers do not cover this; the web says Canberra [W1]."),
    ]}
    mcp = FakeMCP()
    ctx = ctx5("What is the capital of Australia?", s, low, mcp)
    final = await run_request(ctx)
    events = drain(ctx)
    calls = worker_calls(s, "research")
    assert calls[0]["tools"] == ["corpus_search"]
    assert calls[1]["tools"] == ["corpus_search", "tavily_search"]
    gates = ev(events, "tool_gate")
    assert gates[-1]["enabled"] is True and "low corpus support" in gates[-1]["reason"]
    assert gates[-1]["best_corpus_score"] == pytest.approx(0.21)
    assert mcp.calls == ["capital of Australia"]
    assert final["worker_results"]["t1"]["evidence_ids"] == ["W1"]


async def test_needs_live_binds_web_from_round_one():
    s = Script(planner=research_plan(True))
    s.worker_by_agent = {"research": [
        AIMessage(content="", tool_calls=[tool_call("tavily_search", {"query": "agent memory 2026"})]),
        AIMessage(content="- New work on memory [W1]."),
    ]}
    ctx = ctx5("What's new in agent memory?", s, FakeRetriever(), FakeMCP())
    await run_request(ctx)
    events = drain(ctx)
    assert worker_calls(s, "research")[0]["tools"] == ["corpus_search", "tavily_search"]
    plan_ev = ev(events, "plan")[0]
    assert plan_ev["tasks"][0]["needs_live"] is True


async def test_web_failure_gives_partial_corpus_only_research():
    s = Script(planner=research_plan(True))
    s.worker_by_agent = {"research": [
        AIMessage(content="", tool_calls=[tool_call("corpus_search", {"query": "alpha"}),
                                          tool_call("tavily_search", {"query": "alpha 2026"}, 2)]),
        AIMessage(content="- Alpha retrieves when uncertain [C1]. Live web search was unavailable."),
    ]}
    mcp = FakeMCP(TavilyUnavailable("quota", f"Tavily error status=432 tavilyApiKey={FAKE_KEY}"))
    ctx = ctx5("What's new with alpha?", s, FakeRetriever(), mcp)
    final = await run_request(ctx)
    events = drain(ctx)
    r = final["worker_results"]["t1"]
    assert r["status"] == "partial" and r["error_kind"] == "live_search_unavailable"
    assert r["evidence_ids"] == ["C1"]
    tr = [d for d in ev(events, "tool_result") if d["tool"] == "tavily_search"][0]
    assert tr["status"] == "error" and tr["error_kind"] == "quota" and tr["recoverable"] is True
    assert final["status"] == "partial" and ev(events, "done")[0]["status"] == "partial"
    frames = "".join(sse_frame(e, d) for e, d in events)
    assert FAKE_KEY not in frames
    # the model sees a redacted error too
    tool_msgs = [m for m in worker_calls(s, "research")[1]["messages"] if m.type == "tool"]
    assert any("quota" in m.content for m in tool_msgs) and not any(FAKE_KEY in m.content for m in tool_msgs)


async def test_web_disabled_when_no_key_is_partial_not_crash():
    s = Script(planner=research_plan(True))
    s.worker_by_agent = {"research": [
        AIMessage(content="", tool_calls=[tool_call("corpus_search", {"query": "alpha"}),
                                          tool_call("tavily_search", {"query": "alpha"}, 2)]),
        AIMessage(content="- Alpha [C1]."),
    ]}
    ctx = ctx5("alpha news?", s, FakeRetriever(), None)
    final = await run_request(ctx)
    assert final["worker_results"]["t1"]["status"] == "partial"
    assert ev(drain(ctx), "web_search")[0]["error_kind"] == "disabled"


async def test_web_search_budget_per_task():
    ctx = make_ctx("q", Script())
    ctx.tavily = FakeMCP()
    tool = shared_search_tool(ctx, "research", "p")
    await tool.coroutine(query="one")
    await tool.coroutine(query="two")
    with pytest.raises(TavilyUnavailable) as ei:
        await tool.coroutine(query="three")
    assert ei.value.kind == "budget" and ctx.tavily.calls == ["one", "two"]


# ---------------------------------------------------------------- MCP sharing: same tool, two agents


async def test_same_mcp_tool_used_by_research_and_summary_in_different_contexts(monkeypatch):
    import app.agents.summary as summary_mod

    async def fake_outline(paper):
        return "Paper arXiv:2310.11511v1: Self-RAG", []

    plan_ = {"plan_summary": "p", "tasks": [
        {"worker": "research", "instruction": "Find current developments in adaptive retrieval",
         "depends_on": [], "needs_live": True},
        {"worker": "summary", "instruction": "Check the current venue status of Self-RAG",
         "depends_on": [], "needs_live": True},
    ]}
    s = Script(planner=plan_)
    s.worker_by_agent = {
        "research": [
            AIMessage(content="", tool_calls=[tool_call("tavily_search", {"query": "adaptive retrieval 2026"})]),
            AIMessage(content="- Recent adaptive retrieval work [W1]."),
        ],
        "summary": [
            AIMessage(content="", tool_calls=[tool_call("tavily_search", {"query": "Self-RAG ICLR 2024 accepted"})]),
            AIMessage(content="- Self-RAG was published at ICLR 2024 per the web [W3]."),
        ],
    }
    second = [{"url": "https://openreview.net/forum?id=x", "title": "Self-RAG OpenReview", "content": "ICLR 2024",
               "score": 0.9},
              {"url": "https://example.org/a", "title": "A", "content": "dup", "score": 0.1}]
    mcp = FakeMCP(parse_search_result(blocks(GOOD)), second)
    ctx = ctx5("What's new in adaptive retrieval, and was Self-RAG accepted?", s, FakeRetriever(), mcp)
    final = await run_request(ctx)
    events = drain(ctx)

    # one MCP server object served both agents
    assert mcp.calls == ["adaptive retrieval 2026", "Self-RAG ICLR 2024 accepted"]
    ws = ev(events, "web_search")
    assert [w["agent"] for w in ws] == ["research", "summary"]
    assert {w["tool"] for w in ws} == {"tavily_search"} and {w["server"] for w in ws} == {"tavily-remote-mcp"}
    assert ws[0]["purpose"] != ws[1]["purpose"]
    calls = [d for d in ev(events, "tool_call") if d["tool"] == "tavily_search"]
    assert [c["agent"] for c in calls] == ["research", "summary"]
    # Summary keeps its own distinguishing tools next to the shared one; Research keeps corpus_search
    assert set(worker_calls(s, "summary")[0]["tools"]) == {"get_paper_outline", "get_paper_section",
                                                           "search_within_paper", "tavily_search"}
    assert set(worker_calls(s, "research")[0]["tools"]) == {"corpus_search", "tavily_search"}
    # one W# namespace: the URL found by both agents keeps its id
    assert ws[1]["results"][1]["cite_id"] == ws[0]["results"][0]["cite_id"] == "W1"
    assert final["worker_results"]["t2"]["evidence_ids"] == ["W3"]
    assert ctx.registry.get("W3").agent == "summary" and ctx.registry.get("W1").agent == "research"


async def test_summary_without_status_cue_gets_no_web_tool():
    from app.agents.summary import status_cue

    s = Script(planner={"plan_summary": "p", "tasks": [
        {"worker": "summary", "instruction": "Summarise the method of Self-RAG", "depends_on": []}]})
    s.worker_by_agent = {"summary": [AIMessage(content="- nothing")]}
    ctx = ctx5("Summarise the method of Self-RAG", s, FakeRetriever(), FakeMCP())
    await run_request(ctx)
    assert "tavily_search" not in worker_calls(s, "summary")[0]["tools"]
    assert not status_cue("Summarise the method of Self-RAG")
    assert status_cue("Has Self-RAG been accepted at a venue?")


# ---------------------------------------------------------------- Summary tools over the real local index

INDEX = ROOT / "data" / "chroma_index"
DB = ROOT / "data" / "papers.db"
needs_data = pytest.mark.skipif(not (INDEX.exists() and DB.exists()), reason="local index/DB not built")


@pytest.fixture(scope="module")
def real_retriever():
    from app.config import load_settings
    from app.rag import Retriever

    return Retriever(load_settings())


@needs_data
@pytest.mark.parametrize("ref", ["2310.11511", "2310.11511v1", "arXiv:2310.11511", "Self-RAG"])
def test_resolve_paper_by_id_or_title(ref):
    from app.agents.summary import resolve_paper

    assert resolve_paper(DB, ref)["arxiv_id"] == "2310.11511"


@needs_data
@pytest.mark.parametrize("ref,frag", [("9999.99999", "not in the indexed corpus"),
                                      ("Attention Is All You Need", "not in the corpus"),
                                      ("Agent", "matches several papers"),
                                      ("Large Language Model", "matches several papers")])
def test_resolve_paper_unknown_or_ambiguous(ref, frag):
    from app.agents.summary import PaperNotFound, resolve_paper

    with pytest.raises(PaperNotFound, match=frag):
        resolve_paper(DB, ref)


@needs_data
async def test_summary_tools_outline_section_and_cap(real_retriever):
    from app.agents.summary import SECTION_TOKEN_CAP, summary_tools
    from app.llm.policy import approx_tokens

    ctx = make_ctx("q", Script(), real_retriever, sqlite_path=DB)
    tools = {t.name: t for t in summary_tools(ctx)}
    text, ids = await tools["get_paper_outline"].coroutine(paper="Self-RAG")
    assert "Self-RAG" in text and "Sections (section_id" in text and "abstract |" in text
    assert ids and all(ctx.registry.get(i).chunk_id.startswith("2310.11511v1::abstract::") for i in ids)
    sec_id = next(line.split("|")[0].strip("- ").strip() for line in text.splitlines()
                  if line.startswith("- ") and "abstract" not in line)
    body, sec_ids = await tools["get_paper_section"].coroutine(paper="2310.11511", section=sec_id)
    refs = [ctx.registry.get(i) for i in sec_ids]
    assert refs and all(r.chunk_id.split("::")[1] == sec_id for r in refs)
    assert [r.chunk_id for r in refs] == sorted(r.chunk_id for r in refs)  # reading order
    assert sum(approx_tokens(r.text) for r in refs[:-1]) <= SECTION_TOKEN_CAP
    assert tools["get_paper_section"].metadata == {"reads_full_text": True}
    events = drain(ctx)
    assert {d["tool"] for d in ev(events, "retrieval")} == {"get_paper_outline", "get_paper_section"}
    assert all(d["agent"] == "summary" for d in ev(events, "retrieval"))


@needs_data
async def test_get_paper_section_accepts_title_and_rejects_unknown(real_retriever):
    from app.agents.summary import PaperNotFound, summary_tools

    ctx = make_ctx("q", Script(), real_retriever, sqlite_path=DB)
    tools = {t.name: t for t in summary_tools(ctx)}
    _, ids = await tools["get_paper_section"].coroutine(paper="2310.11511", section="Abstract")
    assert ids
    with pytest.raises(PaperNotFound):
        await tools["get_paper_section"].coroutine(paper="2310.11511", section="No Such Section XYZ")


# ---------------------------------------------------------------- worker-side routing (P4-N2)


def test_long_full_section_text_routes_worker_call_to_capability():
    c = policy.choose("worker_tool_loop", {"long_source_tokens_est": 3400, "context_tokens_est": 4200})
    assert c.provider == policy.OPENAI and c.dimension == "capability" and "full paper sections" in c.rationale
    c2 = policy.choose("worker_final", {"long_source_tokens_est": 0, "context_tokens_est": 11000})
    assert c2.provider == policy.OPENAI


def test_short_worker_context_stays_on_cost_side():
    c = policy.choose("worker_tool_loop", {"long_source_tokens_est": 900, "context_tokens_est": 3000})
    assert c.provider == policy.GEMINI and c.dimension == "cost"


async def test_summary_loop_switches_provider_after_reading_long_section():
    from app.agents.worker import run_tool_loop
    from langchain_core.tools import StructuredTool

    async def read(section: str):
        return "word " * 16000  # ~ 20k chars -> ~5k tokens

    tool = StructuredTool.from_function(coroutine=read, name="get_paper_section", description="d",
                                        metadata={"reads_full_text": True})
    s = Script()
    s.worker_by_agent = {}
    ctx = make_ctx("Summarise the method", s)

    class C(FakeChat):
        async def _agenerate(self, messages, stop=None, run_manager=None, **kw):
            n = sum(1 for c in s.calls if c.get("site") == "w")
            s.calls.append({"site": "w"})
            msg = AIMessage(content="", tool_calls=[tool_call("get_paper_section", {"section": "S3"})]) if n == 0 \
                else AIMessage(content="- findings")
            return ChatResult(generations=[ChatGeneration(message=msg)])

    from tests.phase2_fakes import FakeRouter

    class R(FakeRouter):
        def client(self, provider):
            return C(script=s)

    ctx.router = R(ctx.settings, s)
    await run_tool_loop(ctx, agent="summary", system="sys", user="u", tools=[tool], max_rounds=3)
    routes = ev(drain(ctx), "llm_route")
    assert [r["provider"] for r in routes] == [policy.GEMINI, policy.OPENAI]
    assert routes[1]["dimension"] == "capability" and routes[1]["features"]["long_source_tokens_est"] >= 3000


# ---------------------------------------------------------------- P3-N3 abstentions and P2-N8 planner


def test_abstention_only_markers():
    assert abstention_only_markers("The available evidence does not answer this [C1].") == {"C1"}
    assert abstention_only_markers("X does Y [C1]. The evidence does not cover Z [C2].") == {"C2"}
    assert abstention_only_markers("X does Y [C1], but the papers do not mention Z [C1].") == set()
    assert abstention_only_markers("No relevant evidence was found [C1, C2]; see [W1] for news.") == {"C1", "C2"}


async def test_worker_abstention_gives_synthesis_no_unrelated_evidence():
    s = Script(planner=research_plan(False), synth_chunks=["The available evidence does not answer this [C1]."])
    s.worker_by_agent = {"research": [
        AIMessage(content="", tool_calls=[tool_call("corpus_search", {"query": "alpha"})]),
        AIMessage(content="- The available evidence does not answer this."),
    ]}
    ctx = ctx5("Something the corpus lacks?", s, FakeRetriever(), FakeMCP())
    await run_request(ctx)
    events = drain(ctx)
    human = s.synth_messages[-1].content
    assert "(no evidence retrieved)" in human and "Alpha method" not in human
    assert ev(events, "citation") == []
    assert ev(events, "done")[0]["dropped_citations"] == ["C1"]


async def test_planner_failure_fallback_task_needs_live():
    s = Script(planner=RuntimeError("bad planner"))
    s.worker_by_agent = {"research": [AIMessage(content="- nothing")]}
    ctx = ctx5("Who won the 2026 World Cup?", s, FakeRetriever(), FakeMCP())
    await run_request(ctx)
    assert ev(drain(ctx), "plan")[0]["tasks"][0]["needs_live"] is True
    assert "tavily_search" in worker_calls(s, "research")[0]["tools"]


# ---------------------------------------------------------------- empty completions (deployed 2026-10-27)


async def test_empty_completion_is_retried_then_falls_back_not_treated_as_an_answer():
    """Gemini via OpenRouter returned HTTP 200 with no text, tool calls or usage on Cloud Run; the worker
    then 'abstained'. Now: llm_result error empty_response, one retry, then fallback to the other provider."""
    from app.agents.worker import run_tool_loop
    from langchain_core.tools import StructuredTool
    from tests.phase2_fakes import FakeRouter

    async def outline(paper: str):
        return "outline"

    tool = StructuredTool.from_function(coroutine=outline, name="get_paper_outline", description="d")
    s = Script()
    replies = [
        AIMessage(content="", tool_calls=[tool_call("get_paper_outline", {"paper": "Self-RAG"})],
                  usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}),
        AIMessage(content=""),  # empty completion, attempt 1 (Gemini)
        AIMessage(content=""),  # empty completion, retry (Gemini)
        AIMessage(content="- findings [C1]", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}),
    ]

    class C(FakeChat):
        async def _agenerate(self, messages, stop=None, run_manager=None, **kw):
            n = sum(1 for c in s.calls if c.get("site") == "w")
            s.calls.append({"site": "w"})
            return ChatResult(generations=[ChatGeneration(message=replies[n])])

    class R(FakeRouter):
        def client(self, provider):
            return C(script=s)

    ctx = make_ctx("q", s)
    ctx.router = R(ctx.settings, s)
    out = await run_tool_loop(ctx, agent="summary", system="sys", user="u", tools=[tool], max_rounds=3)
    events = drain(ctx)
    assert out.text == "- findings [C1]"
    results = ev(events, "llm_result")
    assert [r["status"] for r in results] == ["ok", "error", "error", "ok"]
    assert results[1]["error_kind"] == results[2]["error_kind"] == "empty_response"
    assert len(ev(events, "llm_retry")) == 1
    fb = ev(events, "fallback")
    assert len(fb) == 1 and fb[0]["from_provider"] == policy.GEMINI and fb[0]["to_provider"] == policy.OPENAI
    assert results[3]["provider"] == policy.OPENAI


def test_tool_call_or_text_is_not_empty():
    from app.llm.router import is_empty_completion

    assert is_empty_completion(AIMessage(content=""))
    assert not is_empty_completion(AIMessage(content="x"))
    assert not is_empty_completion(AIMessage(content="", tool_calls=[tool_call("t", {})]))


# ---------------------------------------------------------------- off-topic filter scope (P5-D1 / P5-D3)


def test_off_topic_filter_uses_strongest_corpus_score_and_spares_in_paper_chunks():
    from app.graph import LOW_RELEVANCE_SCORE
    from app.rag import EvidenceRegistry

    reg = EvidenceRegistry()
    low = chunk("2302.04761v1", "S5", 0, "Toolformer table", score=0.22)
    a = reg.register_chunk(low)
    assert reg.corpus_off_topic(a.cite_id, LOW_RELEVANCE_SCORE)  # only seen by corpus search, low score
    # the same chunk later scores high on a better corpus query: keep it
    reg.register_chunk(chunk("2302.04761v1", "S5", 0, "Toolformer table", score=0.61))
    assert not reg.corpus_off_topic(a.cite_id, LOW_RELEVANCE_SCORE)
    # low on corpus search, then read inside the named paper by Summary: keep it
    b = reg.register_chunk(chunk("2305.16291v1", "S4", 1, "gpt-4-0314 APIs", score=0.21))
    reg.register_chunk(chunk("2305.16291v1", "S4", 1, "gpt-4-0314 APIs", score=1.0), scope="paper")
    assert not reg.corpus_off_topic(b.cite_id, LOW_RELEVANCE_SCORE)
    # a scoped within-paper hit with a low score is never treated as off-topic
    c = reg.register_chunk(chunk("2305.16291v1", "S4", 2, "API cost", score=0.20), scope="paper")
    assert not reg.corpus_off_topic(c.cite_id, LOW_RELEVANCE_SCORE)


async def test_excluded_off_topic_evidence_is_reported_in_done():
    off = [chunk("2302.04761v1", "S5", 0, "Table 9", title="Toolformer", score=0.22)]
    s = Script(planner=research_plan(True), synth_chunks=["About 70 °C [W1]."])
    s.worker_by_agent = {"research": [
        AIMessage(content="", tool_calls=[tool_call("corpus_search", {"query": "everest"}),
                                          tool_call("tavily_search", {"query": "everest boiling"}, 2)]),
        AIMessage(content="- About 70 °C [W1].\n- The papers do not cover this [C1]."),
    ]}
    ctx = ctx5("Everest boiling point?", s, FakeRetriever(lambda q: off), FakeMCP())
    await run_request(ctx)
    events = drain(ctx)
    assert ev(events, "done")[0]["excluded_evidence"] == ["C1"]
    assert "Table 9" not in s.synth_messages[-1].content


# ---------------------------------------------------------------- Phase 9: method names outside titles


@needs_data
@pytest.mark.parametrize("ref,arxiv", [("MMOA-RAG", "2501.15228"), ("HyDE", "2212.10496"), ("IRCoT", "2212.10509"),
                                       ("GraphRAG", "2404.16130"), ("CodeAct", "2402.01030"), ("mmoa-rag", "2501.15228")])
def test_resolve_paper_by_method_name_in_abstract(ref, arxiv):
    from app.agents.summary import resolve_paper

    row = resolve_paper(DB, ref)
    assert row["arxiv_id"] == arxiv and row["resolved_by"] == "abstract"


@needs_data
@pytest.mark.parametrize("ref,frag", [("R1-Searcher", "title or abstract names"),  # only a TeaRAG baseline
                                      ("Voyager-2", "title or abstract names"),     # not Voyager
                                      ("APIBench", "several abstracts name it"),    # Gorilla and ToolLLM
                                      ("hyde", "title or abstract names")])         # short names match case
def test_resolve_paper_abstract_fallback_refuses_unsafe_matches(ref, frag):
    from app.agents.summary import PaperNotFound, resolve_paper

    with pytest.raises(PaperNotFound, match=frag):
        resolve_paper(DB, ref)


@needs_data
def test_title_match_still_wins_and_is_not_marked():
    from app.agents.summary import resolve_paper

    row = resolve_paper(DB, "Self-RAG")
    assert row["arxiv_id"] == "2310.11511" and "resolved_by" not in row


@needs_data
async def test_outline_says_when_the_paper_was_resolved_by_its_abstract(real_retriever):
    from app.agents.summary import summary_tools

    ctx = make_ctx("q", Script(), real_retriever, sqlite_path=DB)
    tools = {t.name: t for t in summary_tools(ctx)}
    text, cites = await tools["get_paper_outline"].coroutine(paper="MMOA-RAG")
    assert "arXiv:2501.15228" in text and "through this paper's abstract" in text and cites
