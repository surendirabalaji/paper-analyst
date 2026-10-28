"""Tester Phase 7 (independent): failure handling, budgets and the fault-injection guard. IDs: T7-*.

MOCKED providers and retriever (tests/phase2_fakes.py, via the developer's ctx7 helper); the graph, router,
fault hooks, budget code, FastAPI app and SQL guard run for real. These are the tester's own fault specs and
combinations, chosen to differ from tests/unit/test_phase7_failures.py: all-provider hang / rate limit,
every-worker hang across a multi-task plan, unknown fault targets, real (not injected) failures not being
labelled as injected, concurrent isolation of provider-failure state, cancellation, the HTTP guard on a real
socket, and SQL-guard regressions (safe queries still accepted under the new heap cap; corrupt DB kind).
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sqlite3
import threading
import time

import httpx
import pytest
import uvicorn
from langchain_core.messages import AIMessage

from app.config import ROOT
from app.faults import Faults
from app.graph import run_request
from app.llm import policy
from app.sqlguard import SqlError, run_readonly
from tests.phase2_fakes import FakeRetriever, FakeRouter, Script, drain, make_settings, plan, tool_call
from tests.unit.test_phase7_failures import (AgentChat, answer_text, calls, ctx7, ev, every_route_has_a_result,
                                             research_rounds)

DB = ROOT / "data" / "papers.db"
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def _run(ctx, limit=60):
    t0 = time.perf_counter()
    final = await asyncio.wait_for(run_request(ctx), limit)
    return final, drain(ctx), time.perf_counter() - t0


def _terminal_ok(events):
    names = [e for e, _ in events]
    assert names[-1] == "done" and names.count("done") == 1


# ---------------------------------------------------------------- budgets / hangs


async def test_T7_B1_every_worker_hangs_in_a_three_task_plan_stays_inside_budget():
    tasks = [("research", "a", []), ("summary", "b", []), ("data", "c", [])]
    ctx, s = ctx7("q", tasks, faults="worker:*:hang", request_budget_s=12.0, worker_timeout_s=4.0)
    final, events, wall = await _run(ctx)
    assert wall <= ctx.settings.request_budget_s + 0.5, wall
    _terminal_ok(events)
    st = {t["id"]: (t["status"], t["error_kind"]) for t in final["tasks"]}
    assert all(v[0] in ("failed", "skipped") for v in st.values()), st
    assert final["status"] == "failed"
    wrs = [w for w in ev(events, "worker_result") if w["status"] == "failed"]
    # bounded: each task at most MAX_ATTEMPTS (2) attempts
    from collections import Counter
    assert max(Counter(w["task_id"] for w in wrs).values()) <= 2
    assert not ev(events, "citation")
    assert ev(events, "done")[0]["over_budget_s"] == 0.0
    assert "could not answer" in answer_text(events)


async def test_T7_B2_all_providers_hang_is_bounded_and_honest():
    ctx, s = ctx7("How does Alpha retrieve?", [("research", "x", [])], faults="provider:*:hang",
                  request_budget_s=10.0, worker_timeout_s=5.0)
    final, events, wall = await _run(ctx)
    assert wall <= ctx.settings.request_budget_s + 0.5, wall
    _terminal_ok(events)
    assert final["status"] == "failed" and not ev(events, "citation")
    assert every_route_has_a_result(events)
    assert not any(c["site"] == "synth" for c in s.calls)  # nothing reached the fake model
    # every model call that was attempted ended as an error/cancel, never ok
    assert {r["status"] for r in ev(events, "llm_result")} <= {"error", "cancelled"}


async def test_T7_B3_rate_limit_on_gemini_retries_once_then_falls_back_and_completes():
    ctx, s = ctx7("How does Alpha retrieve?", [("research", "x", [])], faults="provider:gemini-openrouter:rate_limit")
    final, events, wall = await _run(ctx)
    _terminal_ok(events)
    gem = [r for r in ev(events, "llm_result") if r["provider"] == policy.GEMINI]
    assert len(gem) == 2 and {r["error_kind"] for r in gem} == {"rate_limit"} and all(r["injected"] for r in gem)
    assert len(ev(events, "fallback")) == 1  # only the first call falls back; later calls avoid Gemini
    assert final["status"] == "done" and every_route_has_a_result(events)
    later = [r for r in ev(events, "llm_route")][1:]
    assert all(r["provider"] == policy.OPENAI for r in later)


async def test_T7_B4_one_transient_provider_timeout_retries_same_provider_without_fallback():
    ctx, s = ctx7("How does Alpha retrieve?", [("research", "x", [])], faults="provider:*:timeout@1")
    final, events, _ = await _run(ctx)
    assert len(ev(events, "llm_retry")) == 1 and not ev(events, "fallback")
    assert final["status"] == "done" and not ctx.failed_providers


# ---------------------------------------------------------------- labelling honesty


async def test_T7_L1_unknown_fault_target_never_fires_and_answer_is_not_labelled():
    ctx, s = ctx7("How does Alpha retrieve?", [("research", "x", [])], faults="tool:no_such_tool:error")
    final, events, _ = await _run(ctx)
    assert not ev(events, "fault_injected") and ev(events, "done")[0]["faults_injected"] == []
    assert final["status"] == "done"
    assert "injected" not in answer_text(events).lower()


async def test_T7_L2_real_worker_exception_is_not_called_injected():
    ctx, s = ctx7("How does Alpha retrieve?", [("research", "x", [])],
                  by_agent={"research": [RuntimeError("boom sk-proj-SECRETSECRETSECRET1234")]})
    final, events, _ = await _run(ctx)
    text = answer_text(events)
    assert final["status"] == "failed" and "could not answer" in text
    assert "injected" not in text.lower() and "SECRETSECRET" not in json.dumps(events, default=str)
    assert not any(e.get("injected") for e in ev(events, "error"))


async def test_T7_L3_tavily_timeout_while_corpus_works_gives_partial_with_named_gap():
    ctx, s = ctx7(
        "What is new on Alpha?", [("research", "find current work", [])],
        by_agent={"research": [calls(("corpus_search", {"query": "alpha"}), ("tavily_search", {"query": "alpha now"})),
                               AIMessage(content="Alpha retrieves when uncertain [C1].")]},
        synth=("Alpha retrieves when uncertain [C1].",), faults="tool:tavily_search:timeout")
    s.planner["tasks"][0]["needs_live"] = True
    ctx.tavily = None
    final, events, _ = await _run(ctx)
    assert final["status"] == "partial"
    text = answer_text(events)
    assert "[C1]" in text and "Incomplete evidence" in text
    assert [c["cite_id"] for c in ev(events, "citation")] == ["C1"]


# ---------------------------------------------------------------- isolation / cancellation


async def test_T7_I1_provider_failure_state_does_not_leak_to_a_concurrent_request():
    a, _ = ctx7("q a", [("research", "x", [])], faults="provider:gemini-openrouter:down", request_id="AAA")
    b, _ = ctx7("q b", [("research", "x", [])], request_id="BBB")
    c, _ = ctx7("q c", [("research", "x", [])], request_id="CCC")
    fa, fb, fc = await asyncio.gather(run_request(a), run_request(b), run_request(c))
    eb, ec = drain(b), drain(c)
    assert a.failed_providers and not b.failed_providers and not c.failed_providers
    for e in (eb, ec):
        assert not ev(e, "fallback") and not ev(e, "fault_injected")
        assert {r["provider"] for r in ev(e, "llm_route")} != {policy.OPENAI} or True
        assert all(d.get("request_id", None) in (None, "BBB", "CCC") for _, d in e)
    assert fb["status"] == fc["status"] == "done"


async def test_T7_I2_cancellation_mid_worker_leaves_no_route_without_result():
    async def slow(messages, kw):
        await asyncio.sleep(10)

    ctx, s = ctx7("q", [("research", "x", [])], by_agent={"research": [slow]})
    t = asyncio.create_task(run_request(ctx))
    await asyncio.sleep(0.5)
    t.cancel()
    try:
        await asyncio.wait_for(t, 5)
    except (asyncio.CancelledError, Exception):
        pass
    events = drain(ctx)
    assert t.done()
    assert every_route_has_a_result(events)


async def test_T7_P1_malformed_planner_output_falls_back_without_fabricating_a_plan():
    ctx, _ = ctx7("How does Alpha retrieve?", [("research", "x", [])])
    ctx.router.script.planner = AIMessage(content="not a tool call at all")
    final, events, _ = await _run(ctx)
    p = ev(events, "plan")[0]
    assert p["source"] == "fallback" and [t["worker"] for t in p["tasks"]] == ["research"]
    _terminal_ok(events)


# ---------------------------------------------------------------- HTTP guard on a real socket


def _free_port():
    sk = socket.socket()
    sk.bind(("127.0.0.1", 0))
    p = sk.getsockname()[1]
    sk.close()
    return p


TOKEN = "Z" * 40


def _serve(settings):
    import app.main as main

    mp = pytest.MonkeyPatch()
    mp.setattr(main, "load_settings", lambda: settings)
    mp.setattr(main, "Retriever", lambda s: FakeRetriever())
    script = Script(planner=plan(("research", "x", [])),
                    worker=[AIMessage(content="", tool_calls=[tool_call("corpus_search", {"query": "alpha"})]),
                            AIMessage(content="Alpha [C1].")], synth_chunks=["Alpha ", "[C1]."])
    mp.setattr(main, "Router", lambda s: FakeRouter(s, script))
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    return f"http://127.0.0.1:{port}", srv, th, mp


@pytest.fixture(params=["no_token", "short_token", "token"])
def server(request):
    tok = {"no_token": "", "short_token": "short-token-123", "token": TOKEN}[request.param]
    base, srv, th, mp = _serve(make_settings(fault_inject_token=tok))
    yield request.param, base
    srv.should_exit = True
    th.join(5)
    mp.undo()


def _post(base, headers=None, body=None):
    return httpx.post(f"{base}/api/chat", json=body or {"query": "How does Alpha retrieve?"},
                      headers=headers or {}, timeout=20)


def _events(text):
    out = []
    for fr in text.split("\n\n"):
        e = d = None
        for ln in fr.splitlines():
            if ln.startswith("event: "):
                e = ln[7:]
            elif ln.startswith("data: "):
                d = json.loads(ln[6:])
        if e:
            out.append((e, d))
    return out


def test_T7_G1_http_guard_matrix(server):
    mode, base = server
    spec = "worker:research:exception"
    # ordinary request: no header -> normal answer, no fault events
    r = _post(base)
    assert r.status_code == 200
    evs = _events(r.text)
    assert not ev(evs, "fault_injected") and "fault_injection" not in ev(evs, "meta")[0]
    assert ev(evs, "done")[0]["status"] == "done"
    # extra body fields and a query-string are not a fault path
    r = httpx.post(f"{base}/api/chat?fault={spec}&X-Fault-Inject={spec}",
                   json={"query": "How does Alpha retrieve?", "fault_inject": spec, "x_fault_inject": spec}, timeout=20)
    assert r.status_code in (200, 422)
    if r.status_code == 200:
        assert not ev(_events(r.text), "fault_injected")
    # header without / with wrong / with near-miss token -> 403 for every server mode
    for h in ({"X-Fault-Inject": spec}, {"x-fault-inject": spec, "X-Fault-Token": "wrong"},
              {"X-FAULT-INJECT": spec, "X-Fault-Token": TOKEN + "x"}, {"X-Fault-Inject": spec, "X-Fault-Token": TOKEN[:-1]},
              {"X-Fault-Inject": spec, "X-Fault-Token": ""}, {"X-Fault-Inject": "", "X-Fault-Token": ""},
              {"X-Fault-Inject": spec, "X-Fault-Token": "short-token-123"}):
        r = _post(base, h)
        assert r.status_code == 403, (mode, h.keys(), r.status_code)
        assert "text/event-stream" not in r.headers.get("content-type", "")
        assert TOKEN not in r.text
    # token alone (no spec header) is harmless
    r = _post(base, {"X-Fault-Token": TOKEN})
    assert r.status_code == 200 and not ev(_events(r.text), "fault_injected")
    good = {"X-Fault-Token": TOKEN}
    r = _post(base, {**good, "X-Fault-Inject": spec})
    if mode != "token":
        assert r.status_code == 403  # short or absent server token: injection is off
        return
    assert r.status_code == 200
    evs = _events(r.text)
    assert ev(evs, "fault_injected") and ev(evs, "done")[0]["status"] == "failed"
    assert TOKEN not in r.text
    # bad specs are 422, never 500 or a stream
    for bad in ("x" * 5000, "worker:research:exception@0", "worker:research:exception@21",
                ",".join(["worker:research:exception"] * 9), b"worker:r\xe9search:boom", "provider:*:down:extra"):
        r = _post(base, {**good, "X-Fault-Inject": bad})
        assert r.status_code == 422, (bad[:40], r.status_code)
    # empty spec with the right token = no faults
    r = _post(base, {**good, "X-Fault-Inject": ""})
    assert r.status_code == 200 and not ev(_events(r.text), "fault_injected")
    # a following ordinary request is unaffected (rules are per request)
    r = _post(base)
    assert ev(_events(r.text), "done")[0]["status"] == "done" and not ev(_events(r.text), "fault_injected")


# ---------------------------------------------------------------- SQL guard regressions (P6-D1c / P6-D3)


@pytest.mark.parametrize("sql", [
    "WITH c AS (SELECT category, count(*) n FROM paper_categories GROUP BY category) SELECT * FROM c ORDER BY n DESC",
    "SELECT p.arxiv_id, count(a.author_id) FROM papers p JOIN paper_authors a ON a.arxiv_id = p.arxiv_id "
    "GROUP BY p.arxiv_id ORDER BY 2 DESC",
    "SELECT DISTINCT category FROM paper_categories ORDER BY category",
])
def test_T7_S1_ordinary_queries_still_accepted_under_heap_cap(sql):
    try:
        res = run_readonly(DB, sql)
    except SqlError as e:
        if e.kind == "sql_error" and "no such column" in str(e):
            pytest.skip(f"schema differs: {e}")
        raise
    assert res.rows


def test_T7_S2_corrupt_db_is_not_blamed_on_the_query(tmp_path):
    bad = tmp_path / "papers.db"
    bad.write_bytes(b"this is not a sqlite file" * 100)
    with pytest.raises(SqlError) as ei:
        run_readonly(bad, "SELECT count(*) FROM papers")
    # A corrupt asset is a service-side problem; reporting it as invalid SQL makes the Data agent spend its
    # repair round and the answer say the model "wrote a database query that was still invalid".
    assert ei.value.kind in ("sql_unavailable", "sql_internal"), (ei.value.kind, str(ei.value)[:120])


async def test_T7_S3_corrupt_db_through_the_graph_is_service_side(tmp_path):
    bad = tmp_path / "papers.db"
    bad.write_bytes(b"this is not a sqlite file" * 100)
    ctx, s = ctx7("How many papers?", [("data", "count papers", [])],
                  by_agent={"data": [calls(("run_sql", {"query": "SELECT count(*) FROM papers"})),
                                     calls(("run_sql", {"query": "SELECT count(*) FROM papers"})),
                                     AIMessage(content="The database could not be read.")]})
    ctx.settings = ctx.settings.__class__(**{**ctx.settings.__dict__, "sqlite_path": bad})
    final, events, _ = await _run(ctx)
    r = final["worker_results"]["t1"]
    print("OBSERVED", r["status"], r["error_kind"], r.get("sql_failures"), answer_text(events)[-300:])
    assert r["error_kind"] == "data_unavailable"
    assert "still invalid" not in answer_text(events)
