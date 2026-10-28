"""Phase 7 offline tests: failure handling, budgets and fault injection.

Provider responses are faked (MOCKED, tests/phase2_fakes.py); the graph, router, worker loop, fault hooks
and SQL guard run for real. Covers each failure path the phase names: timeout, exception, empty response,
tool error, both providers down, budget exhaustion, dependency failure, retry then success; plus the fault
spec parser, the per-request guard, labelling of injected faults, and the open notes P2-N7/P4-N1, P5-N1,
P6-D3, P6-D1c and N9.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.messages import AIMessageChunk

from app.config import ROOT
from app.faults import FaultSpecError, Faults, parse_spec
from app.graph import failure_answer, route_next, run_request
from app.llm import policy
from app.sqlguard import SqlError, run_readonly
from tests.phase2_fakes import FakeChat, FakeRouter, Script, drain, make_ctx, plan, tool_call

DB = ROOT / "data" / "papers.db"
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def ev(events, name):
    return [d for e, d in events if e == name]


def calls(*items):
    return AIMessage(content="", tool_calls=[tool_call(n, a, i) for i, (n, a) in enumerate(items, 1)])


def research_rounds(text="Alpha retrieves when uncertain [C1]."):
    return [calls(("corpus_search", {"query": "alpha retrieval"})), AIMessage(content=text)]


class AgentChat(FakeChat):
    """Per-agent scripted worker rounds (agent detected from the bound tool names)."""

    async def _agenerate(self, messages, stop=None, run_manager=None, **kw):
        names = kw.get("tool_names") or []
        s = self.script
        if "Plan" in names:
            return await super()._agenerate(messages, stop, run_manager, **kw)
        agent = "data" if "run_sql" in names else "summary" if "get_paper_outline" in names else "research"
        n = sum(1 for c in s.calls if c["site"] == "worker" and c.get("agent") == agent)
        s.calls.append({"site": "worker", "agent": agent, "tools": names})
        script = s.by_agent.get(agent, [])
        item = script[n] if n < len(script) else AIMessage(content=f"{agent}: no more script")
        from langchain_core.outputs import ChatGeneration, ChatResult

        return ChatResult(generations=[ChatGeneration(message=await self._resolve(item, messages, kw))])


def ctx7(query, tasks, by_agent=None, synth=("Answer from the evidence [C1].",), faults="", **over):
    s = Script(planner=plan(*tasks), synth_chunks=list(synth))
    s.by_agent = by_agent or {"research": research_rounds()}
    ctx = make_ctx(query, s, sqlite_path=DB, **over)

    class R(FakeRouter):
        def client(self, provider):
            return AgentChat(script=self.script)

    ctx.router = R(ctx.settings, s)
    if faults:
        ctx.faults = Faults.from_spec(faults, "test")
    return ctx, s


async def run(ctx, limit=30):
    final = await asyncio.wait_for(run_request(ctx), limit)
    return final, drain(ctx)


def answer_text(events):
    return "".join(d["delta"] for d in ev(events, "token"))


def every_route_has_a_result(events):
    ids = {d["decision_id"] for d in ev(events, "llm_route")} | {d["decision_id"] for d in ev(events, "fallback")}
    return not (ids - {d["decision_id"] for d in ev(events, "llm_result")})


# ---------------------------------------------------------------- fault spec parsing


def test_parse_spec_accepts_rules_and_counts():
    rules = parse_spec("worker:research:timeout@1, tool:tavily_search:error ,provider:*:down")
    assert [(r.kind, r.name, r.mode, r.limit) for r in rules] == [
        ("worker", "research", "timeout", 1), ("tool", "tavily_search", "error", None), ("provider", "*", "down", None)]
    assert parse_spec("") == []


@pytest.mark.parametrize("spec", [
    "research:timeout", "worker:research:explode", "disk:x:full", "worker:research:timeout@0",
    "worker:research:timeout@x", "provider:openai:error", ",".join(["tool:x:error"] * 9),
])
def test_parse_spec_rejects_bad_rules(spec):
    with pytest.raises(FaultSpecError):
        parse_spec(spec)


def test_count_limited_rule_stops_after_n_hits():
    f = Faults.from_spec("worker:research:timeout@2", "test")
    assert [bool(f.take("worker", "research")) for _ in range(3)] == [True, True, False]
    assert f.take("worker", "summary") is None


# ---------------------------------------------------------------- per-request guard (app.main)


def fake_request(headers, token="", env_spec=""):
    state = SimpleNamespace(settings=SimpleNamespace(fault_inject=env_spec), fault_token=token)
    return SimpleNamespace(headers={k.lower(): v for k, v in headers.items()}, app=SimpleNamespace(state=state))


TOKEN = "t" * 32


def test_guard_ordinary_request_gets_no_faults():
    from app.main import request_faults

    assert not request_faults(fake_request({}, token=TOKEN)).active
    assert not request_faults(fake_request({})).active


@pytest.mark.parametrize("headers,token", [
    ({"X-Fault-Inject": "worker:research:exception"}, ""),  # server has no token: injection disabled
    ({"X-Fault-Inject": "worker:research:exception"}, TOKEN),  # no token sent
    ({"X-Fault-Inject": "worker:research:exception", "X-Fault-Token": "wrong"}, TOKEN),
    ({"X-Fault-Inject": "worker:research:exception", "X-Fault-Token": ""}, ""),  # empty == empty must not pass
])
def test_guard_refuses_fault_header_without_the_right_token(headers, token):
    from app.main import request_faults

    with pytest.raises(HTTPException) as e:
        request_faults(fake_request(headers, token=token))
    assert e.value.status_code == 403


def test_guard_accepts_matching_token_and_validates_spec():
    from app.main import request_faults

    f = request_faults(fake_request({"X-Fault-Inject": "worker:research:exception", "X-Fault-Token": TOKEN}, TOKEN))
    assert f.active and f.source == "header" and f.spec() == ["worker:research:exception"]
    with pytest.raises(HTTPException) as e:
        request_faults(fake_request({"X-Fault-Inject": "nonsense", "X-Fault-Token": TOKEN}, TOKEN))
    assert e.value.status_code == 422


def test_deploy_never_copies_process_wide_fault_injection(tmp_path):
    from scripts.env_to_yaml import ALLOWED, read_values

    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-x\nFAULT_INJECT=worker:*:exception\nFAULT_INJECT_TOKEN=abc\n")
    vals = read_values(env)
    assert "FAULT_INJECT" not in ALLOWED and "FAULT_INJECT" not in vals
    assert vals.get("FAULT_INJECT_TOKEN") == "abc"
    src = (ROOT / "scripts" / "deploy.sh").read_text()
    assert "FAULT_INJECT is set" in src


# ---------------------------------------------------------------- worker failures


async def test_retry_then_success_after_injected_worker_timeout():
    ctx, s = ctx7("How does Alpha retrieve?", [("research", "x", [])], faults="worker:research:timeout@1")
    final, events = await run(ctx)
    (hit,) = ev(events, "fault_injected")
    assert hit["target"] == "worker:research:timeout" and hit["injected"] is True and "not a real outage" in hit["note"]
    wrs = ev(events, "worker_result")
    assert [(w["status"], w.get("attempt")) for w in wrs] == [("failed", 1), ("done", 2)]
    assert wrs[0]["injected"] is True and wrs[0]["error_kind"] == "timeout"
    retry = [t for t in ev(events, "transition") if t["reason"].startswith("retry t1")]
    assert retry and "attempt 2 of 2" in retry[0]["reason"]
    assert final["status"] == "done" and "Incomplete evidence" not in answer_text(events)
    assert ev(events, "done")[0]["faults_injected"][0]["target"] == "worker:research:timeout"


async def test_non_transient_worker_exception_is_not_retried_and_answer_is_honest():
    ctx, s = ctx7("How does Alpha retrieve?", [("research", "x", [])], faults="worker:research:exception")
    final, events = await run(ctx)
    wrs = ev(events, "worker_result")
    assert len(wrs) == 1 and wrs[0]["status"] == "failed" and wrs[0]["error_kind"] == "worker_error"
    assert not [t for t in ev(events, "transition") if t["reason"].startswith("retry")]
    err = [e for e in ev(events, "error") if e["scope"] == "worker"][0]
    assert err["injected"] is True and "[injected test fault]" in err["message_safe"]
    text = answer_text(events)
    assert "could not answer" in text and "injected on purpose" in text and "not a real outage" in text
    assert "try again in a moment" not in text  # an injected failure is not presented as a transient outage
    assert not any(c["site"] == "synth" for c in s.calls)
    assert final["status"] == "failed" and ev(events, "done")[0]["status"] == "failed"
    assert [e for e, _ in events][-1] == "done"


async def test_injected_empty_worker_result_is_a_failure_not_a_silent_success():
    ctx, s = ctx7("q", [("research", "x", [])], faults="worker:research:empty")
    final, events = await run(ctx)
    (w,) = ev(events, "worker_result")
    assert (w["status"], w["error_kind"], w["injected"]) == ("failed", "empty_result", True)
    assert not s.calls or not any(c["site"] == "worker" for c in s.calls)  # the real worker never ran
    assert final["status"] == "failed"


async def test_dependency_failure_skips_dependent_with_visible_reason():
    ctx, s = ctx7("Find Alpha papers and summarise them", [("research", "find", []), ("summary", "summarise", [0])],
                  by_agent={"research": research_rounds(), "summary": [AssertionError("summary must not run")]},
                  faults="worker:research:exception")
    final, events = await run(ctx)
    skipped = [w for w in ev(events, "worker_result") if w["status"] == "skipped"]
    assert len(skipped) == 1 and skipped[0]["task_id"] == "t2"
    assert skipped[0]["error_kind"] == "dependency_failed"
    assert "needs t1 (research), which failed" in skipped[0]["reason"]
    assert final["tasks"][1]["reason"] == skipped[0]["reason"]
    assert "Summary (t2) was skipped: it needs t1 (research), which failed with an internal error" in answer_text(events)
    done = ev(events, "done")[0]
    assert [t["status"] for t in done["tasks"]] == ["failed", "skipped"]


async def test_dependent_still_runs_when_dependency_is_partial_and_answer_names_the_gap():
    # Research's web search fails (injected) but its corpus search works: partial, so Summary still runs.
    ctx, s = ctx7(
        "What is new on Alpha, and summarise it", [("research", "find current work", []), ("summary", "summarise", [0])],
        by_agent={"research": [calls(("corpus_search", {"query": "alpha"}), ("tavily_search", {"query": "alpha 2026"})),
                               AIMessage(content="Alpha retrieves when uncertain [C1].")],
                  "summary": [AIMessage(content="Summary: Alpha retrieves when uncertain [C1].")]},
        synth=("Alpha retrieves when uncertain [C1].",), faults="tool:tavily_search:error")
    s.planner["tasks"][0]["needs_live"] = True  # the web tool is available to Research for this task
    ctx.tavily = SimpleNamespace(search=None)  # never reached: the fault fires first
    final, events = await run(ctx)
    final_tasks = {t["id"]: t for t in final["tasks"]}
    assert final_tasks["t1"]["status"] == "partial" and final_tasks["t1"]["error_kind"] == "live_search_unavailable"
    assert final_tasks["t2"]["status"] == "done"  # ran on partial evidence
    tr = [d for d in ev(events, "tool_result") if d["tool"] == "tavily_search"][0]
    assert tr["status"] == "error" and tr["injected"] is True
    text = answer_text(events)
    assert "Incomplete evidence" in text and "live web search" in text and "[injected test fault" in text
    assert final["status"] == "partial" and ev(events, "done")[0]["gaps"][0]["task_id"] == "t1"
    # the synthesizer prompt was told what is missing
    human = s.synth_messages[-1].content
    assert "Missing or incomplete evidence" in human and "Research (t1) is incomplete: it could not use live web search" in human


async def test_tool_error_with_no_evidence_fails_the_worker_with_a_labelled_tool_error():
    ctx, s = ctx7("q", [("research", "x", [])], by_agent={"research": research_rounds(text="Nothing found.")},
                  faults="tool:corpus_search:error")
    final, events = await run(ctx)
    tr = ev(events, "tool_result")[0]
    assert (tr["status"], tr["error_kind"], tr["injected"]) == ("error", "tool_error", True)
    (w,) = ev(events, "worker_result")
    assert (w["status"], w["error_kind"]) == ("failed", "tool_error")
    assert final["status"] == "failed" and ev(events, "citation") == []


async def test_injected_empty_tool_result_is_labelled():
    ctx, s = ctx7("q", [("research", "x", [])], by_agent={"research": research_rounds(text="No evidence.")},
                  faults="tool:corpus_search:empty")
    final, events = await run(ctx)
    tr = ev(events, "tool_result")[0]
    assert tr["n_results"] == 0 and tr["injected"] is True
    assert ctx.retriever.calls == []


# ---------------------------------------------------------------- provider failures


async def test_injected_gemini_empty_response_retries_then_falls_back_to_openai():
    """P5-N1 under fault injection: empty completion -> one retry on Gemini -> fallback to OpenAI."""
    ctx, s = ctx7("How does Alpha retrieve?", [("research", "x", [])], faults="provider:gemini-openrouter:empty")
    final, events = await run(ctx)
    planner = [r for r in ev(events, "llm_result") if r["call_site"] == "planner"]
    assert [(r["provider"], r["status"], r.get("error_kind")) for r in planner] == [
        (policy.GEMINI, "error", "empty_response"), (policy.GEMINI, "error", "empty_response"), (policy.OPENAI, "ok", None)]
    assert all(r.get("injected") for r in planner[:2]) and not planner[2].get("injected")
    (fb,) = [f for f in ev(events, "fallback") if f["call_site"] == "planner"]
    assert fb["injected"] is True and fb["to_provider"] == policy.OPENAI
    assert ev(events, "llm_retry")[0]["injected"] is True
    later = [r for r in ev(events, "llm_route") if r["call_site"] != "planner"]
    assert later and all(r["provider"] == policy.OPENAI and r["dimension"] == "availability" for r in later)
    assert final["status"] == "done" and every_route_has_a_result(events)


async def test_real_empty_stream_is_retried_not_returned_as_an_empty_answer():
    """P5-N1 (streaming): a synthesis stream that ends with no text is empty_response, not a blank answer."""
    ctx, s = ctx7("How does Alpha retrieve?", [("research", "x", [])])
    n = {"i": 0}
    orig = AgentChat._astream

    async def first_empty(self, messages, stop=None, run_manager=None, **kw):
        n["i"] += 1
        if n["i"] == 1:
            yield ChatGenerationChunk(message=AIMessageChunk(content=""))
            return
        async for c in orig(self, messages, stop, run_manager, **kw):
            yield c

    AgentChat._astream = first_empty
    try:
        final, events = await run(ctx)
    finally:
        AgentChat._astream = orig
    synth = [r for r in ev(events, "llm_result") if r["call_site"] == "synthesize"]
    assert synth[0]["status"] == "error" and synth[0]["error_kind"] == "empty_response" and not synth[0].get("injected")
    assert synth[-1]["status"] == "ok" and answer_text(events).startswith("Answer from the evidence")


async def test_both_providers_down_ends_quickly_with_an_honest_failure_and_no_wasted_retry():
    ctx, s = ctx7("How does Alpha retrieve?", [("research", "x", [])], faults="provider:*:down")
    t0 = time.perf_counter()
    final, events = await run(ctx)
    assert time.perf_counter() - t0 < 10
    assert ev(events, "plan")[0]["source"] == "fallback"  # planner failed on both providers
    assert ctx.providers_down()
    wrs = ev(events, "worker_result")
    assert len(wrs) == 1 and wrs[0]["error_kind"] == "provider_unavailable"  # not retried
    notes = [t["reason"] for t in ev(events, "transition") if "not retrying t1" in t["reason"]]
    assert notes and "both generation providers already failed" in notes[0]
    text = answer_text(events)
    assert "could not answer" in text and "injected on purpose" in text
    assert not any(c["site"] == "synth" for c in s.calls)
    assert final["status"] == "failed" and every_route_has_a_result(events)
    assert {r.get("injected") for r in ev(events, "llm_result")} == {True}


async def test_synthesis_failure_before_streaming_returns_the_cited_worker_findings():
    ctx, s = ctx7("How does Alpha retrieve?", [("research", "x", [])])
    orig = AgentChat._astream

    async def broken(self, messages, stop=None, run_manager=None, **kw):
        raise asyncio.TimeoutError()
        yield  # pragma: no cover

    AgentChat._astream = broken
    try:
        final, events = await run(ctx)
    finally:
        AgentChat._astream = orig
    text = answer_text(events)
    assert "final write-up step failed" in text and "Alpha retrieves when uncertain [C1]" in text
    assert [c["cite_id"] for c in ev(events, "citation")] == ["C1"]
    assert final["status"] == "partial"
    assert [e for e in ev(events, "error") if e["scope"] == "synthesize"][0]["recoverable"] is True


# ---------------------------------------------------------------- budgets (P2-N7 / P4-N1)


async def test_hung_worker_is_cut_inside_the_request_budget():
    ctx, s = ctx7("q", [("research", "x", [])], faults="worker:research:hang", request_budget_s=8.0)
    t0 = time.perf_counter()
    final, events = await run(ctx)
    wall = time.perf_counter() - t0
    assert wall <= ctx.settings.request_budget_s + 0.5, wall
    w = ev(events, "worker_result")[0]
    assert (w["status"], w["error_kind"]) == ("failed", "timeout")
    assert any(t["reason"].startswith("budget") for t in ev(events, "transition"))  # no retry: no time left
    assert ev(events, "done")[0]["over_budget_s"] == 0.0


async def test_hung_synthesis_stream_is_capped_by_the_remaining_budget():
    ctx, s = ctx7("How does Alpha retrieve?", [("research", "x", [])], request_budget_s=8.0, llm_timeout_s=60.0)
    s.synth_delay = 30.0
    t0 = time.perf_counter()
    final, events = await run(ctx)
    wall = time.perf_counter() - t0
    assert wall <= ctx.settings.request_budget_s + 0.5, wall
    synth = [r for r in ev(events, "llm_result") if r["call_site"] == "synthesize"]
    assert len(synth) == 1 and synth[0]["error_kind"] == "timeout"  # no retry or fallback without time
    assert final["status"] == "partial" and "final write-up step failed" in answer_text(events)
    assert not ctx.failed_providers  # a deadline cut is not blamed on the provider


async def test_hung_planner_keeps_the_synthesis_reserve():
    ctx, s = ctx7("q", [("research", "x", [])], faults="provider:*:hang@1", request_budget_s=8.0)
    t0 = time.perf_counter()
    final, events = await run(ctx)
    assert time.perf_counter() - t0 <= 8.5
    assert ev(events, "plan")[0]["source"] == "fallback"
    assert final["tasks"][0]["status"] == "skipped" and final["tasks"][0]["error_kind"] == "budget"
    assert "try again in a moment" in answer_text(events)


async def test_attempt_is_not_started_without_time_left():
    from app.llm.router import BudgetExhausted, call_llm

    ctx, s = ctx7("q", [("research", "x", [])], request_budget_s=0.1)
    await asyncio.sleep(0.2)
    with pytest.raises(BudgetExhausted):
        await call_llm(ctx, ctx.route("planner", "planner"), lambda c: c, "hi")
    res = ev(drain(ctx), "llm_result")
    assert res and res[0]["error_kind"] == "budget_exhausted" and not ctx.failed_providers


def test_route_next_reserves_time_and_reports_unrun_tasks():
    tasks = [{"id": "t1", "worker": "research", "depends_on": [], "status": "pending", "attempts": 0}]
    nxt, tid, reason, out = route_next({"tasks": tasks, "step_count": 0}, 6, 4.0, min_worker_s=5.0)
    assert nxt == "synthesize" and reason.startswith("budget") and out[0]["status"] == "skipped"
    assert out[0]["error_kind"] == "budget"


def test_route_next_skips_retry_when_both_providers_are_down():
    t = {"id": "t1", "worker": "research", "depends_on": [], "status": "failed", "attempts": 1,
         "error_kind": "provider_unavailable"}
    nxt, _, reason, _ = route_next({"tasks": [t], "step_count": 1}, 6, 60, providers_down=True)
    assert nxt == "synthesize" and "both generation providers" in reason
    nxt, tid, reason, _ = route_next({"tasks": [dict(t)], "step_count": 1}, 6, 60)
    assert (nxt, tid) == ("research", "t1") and reason.startswith("retry")


# ---------------------------------------------------------------- failure text matches the cause (N9)


def _state(*tasks):
    return {"tasks": [dict(id=f"t{i}", worker=w, status=st, error_kind=k, depends_on=[], **extra)
                      for i, (w, st, k, extra) in enumerate(tasks, 1)]}


def test_failure_text_for_a_refused_query_says_retrying_will_not_help():
    text = failure_answer(_state(("data", "failed", "sql_rejected", {})))
    assert "will probably not help" in text and "refused" in text and "try again in a moment" not in text


def test_failure_text_for_transient_problems_invites_a_retry():
    text = failure_answer(_state(("research", "failed", "timeout", {}), ("summary", "skipped", "dependency_failed",
                                                                        {"reason": "needs t1 (research), which failed"})))
    assert "try again in a moment" in text


def test_failure_text_for_service_side_problems_does_not_blame_the_question():
    text = failure_answer(_state(("data", "failed", "data_unavailable", {})))
    assert "service side" in text and "rephras" in text and "try again in a moment" not in text


# ---------------------------------------------------------------- Data agent and SQL guard (P6-D3, P6-D1c)


async def test_missing_database_is_data_unavailable_and_does_not_use_the_repair_round(tmp_path):
    ctx, s = ctx7("How many papers?", [("data", "count papers", [])],
                  by_agent={"data": [calls(("run_sql", {"query": "SELECT count(*) FROM papers"})),
                                     calls(("run_sql", {"query": "SELECT count(*) FROM papers"})),
                                     AIMessage(content="The database could not be read.")]})
    ctx.settings = ctx.settings.__class__(**{**ctx.settings.__dict__, "sqlite_path": tmp_path / "missing.db"})
    final, events = await run(ctx)
    r = final["worker_results"]["t1"]
    assert (r["status"], r["error_kind"], r["sql_failures"]) == ("failed", "data_unavailable", 0)
    sql = ev(events, "sql_result")
    assert len(sql) == 1 and sql[0]["error_kind"] == "sql_unavailable"  # second call never reached SQLite
    assert not [d for d in ev(events, "llm_route") if d["features"].get("repair_round")]
    assert "service side" in answer_text(events)


def test_crashed_query_process_is_sql_internal_and_logged(monkeypatch):
    def crashed(*a, **kw):
        return subprocess.CompletedProcess(a, 1, stdout="", stderr="Traceback ...\nImportError: boom sk-proj-ABCDEFGHIJKL")

    import app.sqlguard as sg

    monkeypatch.setattr(sg.subprocess, "run", crashed)
    records = []
    h = logging.Handler()
    h.emit = records.append
    sg.log.addHandler(h)  # the paper_analyst logger does not propagate to the root (caplog) handler
    try:
        with pytest.raises(SqlError) as e:
            run_readonly(DB, "SELECT count(*) FROM papers")
    finally:
        sg.log.removeHandler(h)
    assert e.value.kind == "sql_internal" and "simplify" not in str(e.value)
    rec = [r for r in records if r.name == "paper_analyst.sqlguard"]
    assert rec and "returncode=1" in rec[0].getMessage() and "ImportError" in rec[0].getMessage()
    assert "ABCDEFGHIJKL" not in rec[0].getMessage()


def test_killed_query_process_is_still_a_resource_refusal(monkeypatch):
    import app.sqlguard as sg

    monkeypatch.setattr(sg.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, -24, stdout="", stderr=""))
    with pytest.raises(SqlError) as e:
        run_readonly(DB, "SELECT count(*) FROM papers")
    assert e.value.kind == "sql_rejected" and "resource limits" in str(e.value)


R4 = "replace(replace(replace(replace('x','x','xxxxxxxxxx'),'x','xxxxxxxxxx'),'x','xxxxxxxxxx'),'x','xxxxxxxxxx')"


def _derived(i):
    return f"(SELECT s{i} FROM (SELECT replace('xxxxxxxxxx'||{R4},'x','x')||n AS s{i} FROM r WHERE n<=1400) GROUP BY s{i}) t{i}"


def test_many_materialised_subqueries_are_capped_per_query_not_per_file(tmp_path, monkeypatch):
    """P6-D1c: 8 derived tables reached ~200 MB of temp files before; now temp tables live in SQLite's capped
    heap, so the query is refused and nothing is written to the temp directory."""
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    sql = ("WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r WHERE n<1400) SELECT 1 FROM "
           + ", ".join(_derived(i) for i in range(8)) + " LIMIT 1")
    with pytest.raises(SqlError) as e:
        run_readonly(DB, sql)
    assert e.value.kind == "sql_rejected"
    assert run_readonly(DB, "SELECT count(*) FROM papers").rows == [[89]]  # ordinary queries unaffected


# ---------------------------------------------------------------- isolation


async def test_faults_are_per_request_and_do_not_leak_into_concurrent_requests():
    a, _ = ctx7("q a", [("research", "x", [])], faults="worker:research:exception", request_id="AAA")
    b, _ = ctx7("q b", [("research", "x", [])], request_id="BBB")
    fa, fb = await asyncio.gather(run_request(a), run_request(b))
    ea, eb = drain(a), drain(b)
    assert fa["status"] == "failed" and ev(ea, "fault_injected")
    assert fb["status"] == "done" and not ev(eb, "fault_injected") and "fault_injection" not in ev(eb, "meta")[0]
    assert "fault_injection" in ev(ea, "meta")[0]
