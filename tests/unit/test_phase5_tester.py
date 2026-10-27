"""Tester Phase 5 tests (independent of the main agent's test_phase5_tavily_summary.py).

All MOCKED unless stated: LLM providers and the MCP server are faked; one test drives the REAL
langchain-mcp-adapters client against a local HTTP server that echoes the request URL (no network,
no credits). Summary resolver tests read the real local SQLite DB (read-only).

TC-P5-01..03 reproduce P5-D1 (off-topic answer links unrelated corpus chunks; deployed run D3_offtopic).
"""

from __future__ import annotations

import json
import re
import logging
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from langchain_core.messages import AIMessage

from app.agents.research import research_gate
from app.agents.summary import PaperNotFound, resolve_paper, status_cue
from app.config import ROOT
from app.events import safe_error, sse_frame
from app.graph import abstention_only_markers, run_request
from app.llm import policy
from app.mcp_tavily import TavilyMCP, TavilyUnavailable, parse_search_result, shared_search_tool
from tests.phase2_fakes import FakeRetriever, Script, chunk, drain, make_ctx, make_settings, tool_call
from tests.unit.test_phase5_tavily_summary import FakeMCP, ctx5, research_plan

CANARY = "tvly-dev-CANARYp5tester0000aaaa"
DB = ROOT / "data" / "papers.db"

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def ev(events, name):
    return [d for e, d in events if e == name]


# ---------------------------------------------------------------- P5-D1: unrelated chunks linked (deployed D3)

D3_SENTENCE = ("The indexed arXiv papers do not contain physical or barometric data regarding Mount Everest; "
               "thus, all figures are provided from current web findings [C1], [C6], [W1]–[W7].")


def test_tc_p5_01_d3_sentence_markers_attached_to_missing_evidence_statement_not_linked():
    """Deployed run D3_offtopic: C1 (Toolformer table, sim 0.22) and C6 (FireAct prompt, sim 0.17) were cited only
    to say the corpus lacks the data, yet were linked as citations. Expected: C1/C6 not linked."""
    dropped = abstention_only_markers(D3_SENTENCE)
    assert {"C1", "C6"} <= dropped, f"dropped={dropped}"


async def test_tc_p5_02_offtopic_graph_does_not_emit_citation_for_offtopic_chunks():
    """Graph level (MOCKED LLM + MCP): Research searched the corpus (only off-topic chunks, similarity ~0.2), cited
    them while saying the corpus lacks the answer, and the synthesizer reproduced the deployed D3 wording.
    Property: a chunk cited only to state that evidence is missing must not appear as a citation/source."""
    off = [chunk("2302.04761v1", "S5", 0, "Table 9: Toolformer results on T-REx", title="Toolformer", score=0.22)]
    s = Script(planner=research_plan(True), synth_chunks=[
        "Water boils at about 70 °C on the summit [W1]. ", D3_SENTENCE.replace("[C6], ", "").replace("–[W7]", "")])
    s.worker_by_agent = {"research": [
        AIMessage(content="", tool_calls=[tool_call("tavily_search", {"query": "everest boiling point"})]),
        AIMessage(content="", tool_calls=[tool_call("corpus_search", {"query": "everest boiling point"}, 2)]),
        AIMessage(content="- About 70 °C at the summit [W1].\n- The indexed papers do not contain this data [C1]."),
    ]}
    ctx = ctx5("At what temperature does water boil on Everest?", s, FakeRetriever(lambda q: off), FakeMCP())
    await run_request(ctx)
    events = drain(ctx)
    linked = {c["cite_id"] for c in ev(events, "citation")}
    assert "W1" in linked
    assert "C1" not in linked, f"off-topic chunk linked as a source: {linked}"


def test_tc_p5_03_backstop_regressions_still_hold():
    """Guards the documented backstop cases so a P5-D1 fix does not unlink supported claims."""
    assert abstention_only_markers("Self-RAG trains a critic model [C2]. The papers do not cover costs [C3].") == {"C3"}
    assert abstention_only_markers("Voyager uses GPT-4 via blackbox queries [C1].") == set()
    assert abstention_only_markers("") == set()


def test_tc_p5_04_supported_negative_claim_is_not_unlinked():
    """A paper-supported negative claim is still a supported claim; dropping its link loses grounding."""
    assert abstention_only_markers("Standard RAG cannot address noisy retrieved passages, the paper argues [C2].") == set()


# ---------------------------------------------------------------- Summary status cue (web gate) and resolver


@pytest.mark.parametrize("instruction", [
    "Summarize the Reflexion paper's method and results with citations.",
    "Summarize the Search-o1 paper (2025) section by section.",
    "Explain how current retrieval approaches are described in the Self-RAG paper.",
])
def test_tc_p5_05_plain_summary_instructions_do_not_enable_web(instruction):
    assert status_cue(instruction) is False


@pytest.mark.parametrize("instruction", [
    "Check whether the MetaGPT paper was accepted at a conference.",
    "Has Self-RAG been published at a peer-reviewed venue?",
    "Is there a newer version or notable follow-up work of ReAct?",
])
def test_tc_p5_06_status_instructions_enable_web(instruction):
    assert status_cue(instruction) is True


@pytest.mark.parametrize("ref,arxiv", [("2310.11511", "2310.11511"), ("2310.11511v1", "2310.11511"),
                                       ("arXiv:2302.04761", "2302.04761"), ("Toolformer", "2302.04761")])
def test_tc_p5_07_resolver_ids_and_titles(ref, arxiv):
    assert resolve_paper(DB, ref)["arxiv_id"] == arxiv


@pytest.mark.parametrize("ref", ["%%%", "___", "' OR 1=1 --", "x'; DROP TABLE papers; --", "1706.03762",
                                 "Attention Is All You Need", "Agent"])
def test_tc_p5_08_resolver_rejects_wildcards_injection_unknown_and_ambiguous(ref):
    with pytest.raises(PaperNotFound):
        resolve_paper(DB, ref)
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    assert con.execute("select count(*) from papers").fetchone()[0] > 50  # table intact
    con.close()


def test_tc_p5_09_resolver_connection_is_read_only():
    from app.agents.summary import _connect

    con = _connect(DB)
    with pytest.raises(sqlite3.OperationalError):
        con.execute("CREATE TABLE p5_tester_probe(x)")
    con.close()


# ---------------------------------------------------------------- Research gate and worker routing boundaries


class _Em:
    def __init__(self):
        self.events = []

    def emit(self, name, **d):
        self.events.append((name, d))


class _Ctx:
    def __init__(self, floor=0.55):
        self.emitter = _Em()
        self.settings = make_settings(rag_min_score=floor)


@pytest.mark.parametrize("score,enabled", [(None, False), (0.55, False), (0.81, False), (0.5499, True), (0.18, True)])
def test_tc_p5_10_research_gate_threshold_boundary(score, enabled):
    ctx = _Ctx()
    corpus, web = object(), object()
    gate = research_gate(ctx, False, {"score": score}, corpus, web)
    tools, note = gate(None)
    assert (web in tools) is enabled and (note is not None) is enabled
    tools2, note2 = gate(None)  # unlocked once, stays unlocked, traced once
    assert (web in tools2) is enabled and note2 is None
    assert sum(1 for n, d in ctx.emitter.events if n == "tool_gate" and d["enabled"]) == (1 if enabled else 0)


def test_tc_p5_11_needs_live_enables_from_round_one():
    ctx = _Ctx()
    web = object()
    tools, note = research_gate(ctx, True, {"score": None}, object(), web)(None)
    assert web in tools and note is None


@pytest.mark.parametrize("src,ctxt,dim", [(2999, 9999, "cost"), (3000, 0, "capability"), (0, 10000, "capability")])
def test_tc_p5_12_worker_capability_switch_boundaries(src, ctxt, dim):
    for site in ("worker_tool_loop", "worker_final"):
        c = policy.base_choice(site, {"long_source_tokens_est": src, "context_tokens_est": ctxt})
        assert c.dimension == dim, (site, src, ctxt, c)
        assert (c.provider == policy.OPENAI) is (dim == "capability")


# ---------------------------------------------------------------- tavily_search parsing and sharing


def test_tc_p5_13_parse_rejects_non_http_urls_and_classifies_quota():
    bad = {"results": [{"url": "javascript:alert(1)", "title": "x", "content": "y"},
                       {"url": "file:///etc/passwd", "title": "x", "content": "y"}]}
    with pytest.raises(TavilyUnavailable) as e:
        parse_search_result([{"type": "text", "text": json.dumps(bad)}])
    assert e.value.kind == "empty"
    for st, kind in ((432, "quota"), (433, "quota"), (429, "rate_limit"), (403, "auth"), (500, "upstream_error")):
        with pytest.raises(TavilyUnavailable) as e:
            parse_search_result([{"type": "text", "text": json.dumps({"error": "x", "status": st})}])
        assert e.value.kind == kind
    with pytest.raises(TavilyUnavailable) as e:
        parse_search_result(("ignored", {"structured_content": {"results": None}}))
    assert e.value.kind == "empty"


async def test_tc_p5_14_two_agents_same_tool_distinct_purposes_shared_w_namespace_and_budget():
    mcp = FakeMCP()
    ctx = make_ctx("q", Script())
    ctx.tavily = mcp
    r = shared_search_tool(ctx, "research", "find current developments on the topic")
    s = shared_search_tool(ctx, "summary", "check the current status of a specific paper")
    assert r.name == s.name == "tavily_search"
    _, ids_r = await r.coroutine(query="agent frameworks 2026")
    _, ids_s = await s.coroutine(query="MetaGPT venue")
    assert ids_r == ids_s == ["W1", "W2"]  # same URLs -> same W# ids; found_by stays the first agent
    assert ctx.registry.get("W1").agent == "research"
    await s.coroutine(query="MetaGPT ICLR")
    with pytest.raises(TavilyUnavailable) as e:
        await s.coroutine(query="third call")
    assert e.value.kind == "budget" and len(mcp.calls) == 3  # third summary call never reached MCP
    ws = ev(drain(ctx), "web_search")
    assert [(w["agent"], w["status"]) for w in ws] == [("research", "ok"), ("summary", "ok"), ("summary", "ok"),
                                                       ("summary", "error")]
    assert {w["purpose"] for w in ws} == {"find current developments on the topic",
                                          "check the current status of a specific paper"}
    assert all(w["server"] == "tavily-remote-mcp" and w["transport"] == "streamable_http" for w in ws)


# ---------------------------------------------------------------- credential canaries (REAL MCP client, local server)


class _EchoHandler(BaseHTTPRequestHandler):
    """Answers every MCP request with HTTP 500 and echoes the full request path (including the query string)."""

    def _reply(self):
        body = json.dumps({"error": f"boom for {self.path}", "detail": {"nested": self.path}}).encode()
        self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Echo-Path", self.path)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_POST = do_GET = do_DELETE = _reply

    def log_message(self, *a):
        pass


@pytest.fixture
def echo_server():
    srv = HTTPServer(("127.0.0.1", 0), _EchoHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/mcp/"
    srv.shutdown()


async def test_tc_p5_15_real_mcp_client_http_500_echoing_url_never_leaks_key(echo_server, caplog):
    import app.main  # noqa: F401  (production logger levels and redaction filters)

    m = TavilyMCP(make_settings(tavily_api_key=CANARY, tavily_mcp_base_url=echo_server, tavily_timeout_s=5))
    ctx = make_ctx("q", Script())
    ctx.tavily = m
    tool = shared_search_tool(ctx, "summary", "check the current status of a specific paper")
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(TavilyUnavailable) as ei:
            await tool.coroutine(query="MetaGPT venue")
    frames = "".join(sse_frame(e, d) for e, d in drain(ctx))
    blobs = {"exception": str(ei.value), "sse": frames, "log": caplog.text,
             "safe_error": safe_error(ei.value), "repr": repr(ei.value)}
    leaks = {k for k, v in blobs.items() if CANARY in v or "CANARYp5tester" in v}
    assert leaks == set(), f"canary leaked in {leaks}"
    assert ei.value.kind in ("upstream_error", "connection")
    assert '"status": "error"' in frames and '"recoverable": true' in frames


def test_tc_p5_16_nested_exception_group_and_encoded_forms_redacted():
    url = f"https://mcp.tavily.com/mcp/?tavilyApiKey={CANARY}"
    inner = RuntimeError(f"Client error '401 Unauthorized' for url '{url}'")
    grp = ExceptionGroup("unhandled errors in a TaskGroup", [ExceptionGroup("inner", [inner])])
    for e in (grp, ValueError(url.replace("=", "%3D")), ValueError(f"key {CANARY} rejected")):
        assert "CANARYp5tester" not in safe_error(e), safe_error(e)


# ---------------------------------------------------------------- P5-D2: RedactingFilter vs uvicorn access log


def test_tc_p5_17_redacting_filter_keeps_uvicorn_access_records_formattable():
    """Deployed rev 00005: every 'POST /api/chat' access line produced '--- Logging error --- TypeError: cannot
    unpack non-iterable NoneType object' (48 of 48 requests), because RedactingFilter sets record.args=None and
    uvicorn's AccessFormatter unpacks record.args. Expected: the access line is formatted (and still redacted)."""
    from uvicorn.logging import AccessFormatter

    from app.events import RedactingFilter

    h = logging.Handler()
    h.setFormatter(AccessFormatter('%(client_addr)s - "%(request_line)s" %(status_code)s'))
    h.addFilter(RedactingFilter())
    rec = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
                            ("1.2.3.4:5", "GET", f"/x?tavilyApiKey={CANARY}", "1.1", 200), None)
    assert h.filter(rec)
    line = h.format(rec)  # raises TypeError on rev 00005 code
    assert "GET" in line and "200" in line and "CANARYp5tester" not in line


# ---------------------------------------------------------------- P5-D3 (retest 20261027T1810Z): P5-D1 fix regression


class ScopedRetriever(FakeRetriever):
    """search() also accepts doc_id (Summary's search_within_paper); returns an in-paper chunk with the low
    similarity that real within-paper search produces for short keyword queries (deployed R5: 0.20-0.29)."""

    async def search(self, query: str, k: int, doc_id: str | None = None):
        return await super().search(query, k)


VOY_SETUP = chunk("2305.16291v2", "S3.SS1", 0, "We leverage OpenAI's gpt-4-0314 and gpt-3.5-turbo-0301 APIs for text "
                  "completion.", title="Voyager", score=0.2912)


@pytest.mark.skipif(not DB.exists(), reason="local SQLite metadata DB not built")
async def test_tc_p5_18_in_paper_chunk_from_search_within_paper_is_not_dropped_for_low_global_score():
    """Deployed rev 00006, R5_voyager_rerun: Summary's search_within_paper (scoped to Voyager) returned the
    Experimental Setup chunk (gpt-4-0314 APIs) at similarity 0.29; the worker cited it, but synthesis never saw it
    because the P5-D1 filter drops any C# < 0.35. Scoped searches score lower than corpus-wide ones, so the
    0.35 calibration (off-topic corpus-wide 0.16-0.26) does not transfer. Expected: a chunk from the paper the
    task named, cited by the worker for a supported claim, is given to synthesis and linked."""
    s = Script(planner={"plan_summary": "p", "tasks": [
        {"worker": "summary", "instruction": "Which models does the Voyager paper (2305.16291) use?",
         "depends_on": []}]},
        synth_chunks=["Voyager queries OpenAI's gpt-4-0314 and gpt-3.5-turbo-0301 APIs [C1]."])
    s.worker_by_agent = {"summary": [
        AIMessage(content="", tool_calls=[tool_call("search_within_paper",
                                                    {"paper": "2305.16291", "query": "API models used"})]),
        AIMessage(content="- Voyager uses gpt-4-0314 and gpt-3.5-turbo-0301 via API [C1]."),
    ]}
    ctx = ctx5("Which models does Voyager use?", s, ScopedRetriever(lambda q: [VOY_SETUP]), FakeMCP(),
               sqlite_path=DB)
    final = await run_request(ctx)
    events = drain(ctx)
    assert final["worker_results"]["t1"]["evidence_ids"] == ["C1"]
    linked = {c["cite_id"] for c in ev(events, "citation")}
    assert "C1" in linked, f"supported in-paper citation dropped: linked={linked}, done={ev(events, 'done')}"


# ---------------------------------------------------------------- P5-D3 retest (20261027T1840Z): mixed request


class DocScopedRetriever(FakeRetriever):
    """Corpus-wide search returns an off-topic chunk; a doc-scoped search returns the in-paper chunk."""

    def __init__(self, corpus_hits, paper_hits):
        super().__init__(lambda q: corpus_hits)
        self.paper_hits = paper_hits

    async def search(self, query: str, k: int, doc_id: str | None = None):
        if doc_id:
            for c in self.paper_hits:
                self.known[c.chunk_id] = c
            return list(self.paper_hits)
        return await super().search(query, k)


@pytest.mark.skipif(not DB.exists(), reason="local SQLite metadata DB not built")
async def test_tc_p5_19_mixed_request_excludes_corpus_offtopic_but_links_low_scoring_in_paper_chunk():
    """One request, two workers (MOCKED LLM + MCP). Research cites a corpus-wide hit at 0.22 (off-topic); Summary
    cites a within-paper hit at 0.29. Expected: the Research chunk is withheld from synthesis and listed in
    done.excluded_evidence; the Summary chunk reaches synthesis and is linked; no chunk is both linked and excluded."""
    off = chunk("2302.04761v1", "S5", 0, "Table 9: Toolformer results on T-REx", title="Toolformer", score=0.22)
    s = Script(planner={"plan_summary": "p", "tasks": [
        {"worker": "research", "instruction": "find it", "depends_on": [], "needs_live": False},
        {"worker": "summary", "instruction": "Which models does the Voyager paper (2305.16291) use?",
         "depends_on": []}]},
        synth_chunks=["Voyager queries gpt-4-0314 [C2]. The papers do not cover the other part [C1]."])
    s.worker_by_agent = {
        "research": [AIMessage(content="", tool_calls=[tool_call("corpus_search", {"query": "everest boiling"})]),
                     AIMessage(content="- The papers do not cover this [C1].")],
        "summary": [AIMessage(content="", tool_calls=[tool_call("search_within_paper",
                                                                {"paper": "2305.16291", "query": "API models"}, 2)]),
                    AIMessage(content="- Voyager uses gpt-4-0314 [C2].")],
    }
    ctx = ctx5("Which models does Voyager use, and what else?", s, DocScopedRetriever([off], [VOY_SETUP]),
               FakeMCP(), sqlite_path=DB)
    final = await run_request(ctx)
    events = drain(ctx)
    ids = {t: r.get("evidence_ids") for t, r in final["worker_results"].items()}
    by_text = {ref.cite_id: ref.text for ref in ctx.registry.by_cite.values()}
    off_id = next(c for c, t in by_text.items() if t.startswith("Table 9"))
    in_id = next(c for c, t in by_text.items() if "gpt-4-0314" in t)
    done = ev(events, "done")[0]
    linked = {c["cite_id"] for c in ev(events, "citation")}
    assert done["excluded_evidence"] == [off_id], (done, ids)
    assert "Table 9" not in s.synth_messages[-1].content
    assert "gpt-4-0314 and gpt-3.5" in s.synth_messages[-1].content
    assert (off_id, in_id) == ("C1", "C2"), by_text  # synth_chunks cite these markers
    assert in_id in linked, (linked, done)
    assert off_id not in linked
    assert not (linked & set(done["excluded_evidence"]))


def cited_markers(script):
    return set(re.findall(r"\[([CW]\d+)\]", "".join(script.synth_chunks)))
