"""Phase 8b offline tests (latency package B): planner seed hints, seeded first tool calls, speculative
retrieval, evidence-only single-task plans, per-call-site Gemini reasoning effort, and the synthesis
routing rule for raw section text.

Provider responses and the MCP server are faked (MOCKED). Summary tests read the real local Chroma index and
SQLite DB (no network, no embedding calls).
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from app.graph import Plan, PlannedTask, plan_seeds, run_request, validate_plan
from app.llm import policy
from app.llm.policy import GEMINI, OPENAI
from app.llm.router import Router
from tests.phase2_fakes import FakeRetriever, Script, chunk, drain, make_ctx, make_settings, tool_call
from tests.unit.test_phase5_tavily_summary import DB, FakeMCP, ctx5, needs_data, real_retriever, worker_calls  # noqa: F401

pytestmark = pytest.mark.anyio

ON = dict(planner_seeds=True, evidence_only_single_task=True, speculative_retrieval=True)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def ev(events, name):
    return [d for e, d in events if e == name]


def research_task(queries=(), needs_live=False):
    return {"worker": "research", "instruction": "find it", "depends_on": [], "needs_live": needs_live,
            "queries": list(queries)}


# ---------------------------------------------------------------- planner hints


def test_validate_plan_keeps_hints_only_for_the_worker_that_uses_them():
    tasks, _ = validate_plan(Plan(plan_summary="p", tasks=[
        PlannedTask(worker="summary", instruction="s", papers=["Self-RAG", " Self-RAG ", "x", "ReAct", "CRAG"],
                    queries=["ignored"]),
        PlannedTask(worker="research", instruction="r", queries=["a b", "c d", "e f"], papers=["ignored"]),
        PlannedTask(worker="data", instruction="d", papers=["p1"], queries=["q1"]),
    ]), "q")
    assert tasks[0]["papers"] == ["Self-RAG", "ReAct"] and tasks[0]["queries"] == []  # deduped, too short dropped, cap 2
    assert tasks[1]["queries"] == ["a b", "c d"] and tasks[1]["papers"] == []
    assert tasks[2]["papers"] == [] and tasks[2]["queries"] == []


@needs_data
async def test_plan_seeds_resolve_papers_and_drop_unknown_ones_with_a_note():
    ctx = make_ctx("q", Script(), sqlite_path=DB, **ON)
    tasks = [{"id": "t1", "worker": "summary", "papers": ["Self-RAG", "Attention Is All You Need"]}]
    notes: list[str] = []
    await plan_seeds(ctx, tasks, notes)
    assert tasks[0]["seeds"] == [{"name": "get_paper_outline", "args": {"paper": "2310.11511"}, "by": "planner"}]
    assert len(notes) == 1 and "not seeded" in notes[0]


async def test_plan_seeds_off_means_no_seeds():
    ctx = make_ctx("q", Script())  # dataclass defaults: all Phase 8b switches off
    tasks = [{"id": "t1", "worker": "research", "queries": ["alpha"]}]
    await plan_seeds(ctx, tasks, [])
    assert tasks[0]["seeds"] == []


# ---------------------------------------------------------------- research: speculative + evidence-only


async def test_single_research_task_uses_speculative_search_and_skips_the_write_up():
    s = Script(planner={"plan_summary": "p", "tasks": [research_task(["alpha method"])]},
               worker=[AIMessage(content="DONE")], synth_chunks=["Alpha retrieves when uncertain [C1]."])
    retr = FakeRetriever()
    ctx = make_ctx("How does alpha work?", s, retriever=retr, **ON)
    final = await run_request(ctx)
    events = drain(ctx)
    # The question was searched once (speculatively, during planning) and the planner's query once.
    assert [q for q, _ in retr.calls] == ["How does alpha work?", "alpha method"]
    spec = ev(events, "speculative_retrieval")
    assert len(spec) == 1 and spec[0]["status"] == "kept"
    calls = ev(events, "tool_call")
    assert [c["seeded_by"] for c in calls] == ["speculative", "planner"]
    assert any(r.get("speculative") for r in ev(events, "retrieval"))
    # One LLM worker round (it replied DONE), no worker_final, then synthesis from the gathered evidence.
    sites = [r["call_site"] for r in ev(events, "llm_route")]
    assert sites == ["planner", "worker_tool_loop", "synthesize"]
    wr = ev(events, "worker_result")[0]
    assert wr["status"] == "done" and wr["evidence_only"] is True and wr["evidence_ids"] == ["C1"]
    assert final["status"] == "done"
    human = s.synth_messages[-1].content
    assert "evidence-only task" in human and 'id="C1"' in human
    assert [d["cite_id"] for d in ev(events, "citation")] == ["C1"]


async def test_evidence_only_round_runs_its_batched_calls_then_stops():
    two = AIMessage(content="", tool_calls=[tool_call("corpus_search", {"query": "beta"}, 1),
                                            tool_call("corpus_search", {"query": "gamma"}, 2)])
    s = Script(planner={"plan_summary": "p", "tasks": [research_task(["alpha"])]},
               worker=[two, AIMessage(content="must not be called")], synth_chunks=["ok [C1]"])
    retr = FakeRetriever()
    ctx = make_ctx("How does alpha work?", s, retriever=retr, **ON)
    await run_request(ctx)
    events = drain(ctx)
    assert sum(1 for c in s.calls if c["site"] == "worker") == 1  # no second round, no write-up
    assert [c["tool_choice"] for c in s.calls if c["site"] == "worker"] == ["auto"]  # seeds ran, so not required
    assert len(ev(events, "tool_call")) == 4  # 2 seeds + 2 batched; within MAX_CALLS_PER_TASK


async def test_multi_task_plan_discards_speculation_and_keeps_worker_findings():
    s = Script(planner={"plan_summary": "p", "tasks": [research_task(["alpha"]), research_task(["beta"])]},
               worker=[AIMessage(content="- alpha finding [C1]."), AIMessage(content="- beta finding [C1].")],
               synth_chunks=["ok [C1]"])
    retr = FakeRetriever()
    ctx = make_ctx("Compare alpha and beta", s, retriever=retr, **ON)
    await run_request(ctx)
    events = drain(ctx)
    spec = ev(events, "speculative_retrieval")
    assert len(spec) == 1 and spec[0]["status"] == "discarded"
    assert all(c.get("seeded_by") == "planner" for c in ev(events, "tool_call"))
    assert not any(w.get("evidence_only") for w in ev(events, "worker_result"))
    assert "alpha finding" in s.synth_messages[-1].content  # findings mode: synthesis reconciles written findings


async def test_off_topic_evidence_is_still_withheld_in_evidence_only_mode():
    low = FakeRetriever(lambda q: [chunk("2401.00001v1", "S1", 0, "Unrelated text.", score=0.2)])
    s = Script(planner={"plan_summary": "p", "tasks": [research_task()]}, worker=[AIMessage(content="DONE")],
               synth_chunks=["The indexed papers do not cover this."])
    ctx = make_ctx("What is the capital of Australia?", s, retriever=low, **ON)
    final = await run_request(ctx)
    assert final["excluded_evidence"] == ["C1"]
    assert 'id="C1"' not in s.synth_messages[-1].content


async def test_speculative_search_failure_falls_back_to_a_normal_search():
    class FlakyOnce(FakeRetriever):
        async def search(self, query, k):
            self.calls.append((query, k))
            if len(self.calls) == 1:
                raise RuntimeError("embedding call failed")
            return [chunk("2401.00001v1", "S1", 0, "Alpha retrieves when uncertain.")]

    retr = FlakyOnce()
    s = Script(planner={"plan_summary": "p", "tasks": [research_task()]}, worker=[AIMessage(content="DONE")],
               synth_chunks=["ok [C1]"])
    ctx = make_ctx("How does alpha work?", s, retriever=retr, **ON)
    final = await run_request(ctx)
    assert len(retr.calls) == 2 and final["status"] == "done"


async def test_question_too_long_for_corpus_search_is_not_searched_speculatively():
    retr = FakeRetriever()
    s = Script(planner={"plan_summary": "p", "tasks": [research_task(["alpha"])]}, worker=[AIMessage(content="DONE")],
               synth_chunks=["ok [C1]"])
    ctx = make_ctx("alpha " * 80, s, retriever=retr, **ON)
    final = await run_request(ctx)
    events = drain(ctx)
    assert ev(events, "speculative_retrieval") == [] and [q for q, _ in retr.calls] == ["alpha"]
    assert final["status"] == "done"


# ---------------------------------------------------------------- summary: seeded outline + evidence-only


@needs_data
async def test_single_summary_task_seeds_outline_reads_sections_in_one_round(real_retriever):  # noqa: F811
    s = Script(planner={"plan_summary": "p", "tasks": [
        {"worker": "summary", "instruction": "Summarise Self-RAG", "depends_on": [], "papers": ["Self-RAG"]}]},
        synth_chunks=["Self-RAG trains reflection tokens [C1]."])
    s.worker_by_agent = {"summary": [
        AIMessage(content="", tool_calls=[tool_call("get_paper_section", {"paper": "2310.11511", "section": "S1"}, 1),
                                          tool_call("get_paper_section", {"paper": "2310.11511", "section": "S3"}, 2)]),
        AIMessage(content="must not be called"),
    ]}
    ctx = ctx5("Summarize Self-RAG", s, real_retriever, FakeMCP(), sqlite_path=DB, **ON)
    await run_request(ctx)
    events = drain(ctx)
    calls = ev(events, "tool_call")
    assert calls[0]["tool"] == "get_paper_outline" and calls[0]["seeded_by"] == "planner"
    assert [c["tool"] for c in calls[1:]] == ["get_paper_section", "get_paper_section"]
    assert len(worker_calls(s, "summary")) == 1
    assert "Evidence already gathered" in worker_calls(s, "summary")[0]["messages"][1].content
    wr = ev(events, "worker_result")[0]
    assert wr["evidence_only"] is True and len(wr["evidence_ids"]) > 2
    synth = [r for r in ev(events, "llm_route") if r["call_site"] == "synthesize"][0]
    assert synth["features"]["evidence_only"] is True


# ---------------------------------------------------------------- routing


def test_evidence_only_synthesis_keeps_the_phase4_routing_table():
    # One short single-paper answer stays on the cost side; long evidence or many papers go to capability.
    assert policy.base_choice("synthesize", {"n_tasks_ok": 1, "n_papers": 1, "evidence_tokens_est": 4200,
                                             "evidence_only": True}).provider == GEMINI
    assert policy.base_choice("synthesize", {"n_tasks_ok": 1, "n_papers": 1, "evidence_tokens_est": 7000,
                                             "evidence_only": True}).provider == OPENAI
    assert policy.base_choice("synthesize", {"n_tasks_ok": 1, "n_papers": 4, "evidence_tokens_est": 3000,
                                             "evidence_only": True}).provider == OPENAI


def test_gemini_effort_is_minimal_except_the_planner():
    r = Router(make_settings(gemini_reasoning_effort="minimal", gemini_planner_reasoning_effort="low",
                             openai_reasoning_effort="", openrouter_api_key="sk-or-FAKEFAKEFAKE"))
    plan_d = r.decide("planner", "planner", {})
    work_d = r.decide("worker_tool_loop", "research", {})
    assert plan_d.provider == GEMINI and plan_d.reasoning_effort == "low"
    assert work_d.provider == GEMINI and work_d.reasoning_effort == "minimal"
    assert r.client_for(plan_d) is not r.client_for(work_d)
    assert r.client_for(work_d) is r.client(GEMINI)
    assert r.client_for(plan_d).extra_body["reasoning"] == {"effort": "low"}
    assert r.client(GEMINI).extra_body["reasoning"] == {"effort": "minimal"}


def test_load_settings_turns_package_b_on_by_default(monkeypatch):
    from app.config import load_settings

    for k in ("PLANNER_SEEDS", "EVIDENCE_ONLY_SINGLE_TASK", "SPECULATIVE_RETRIEVAL", "GEMINI_REASONING_EFFORT",
              "GEMINI_PLANNER_REASONING_EFFORT"):
        monkeypatch.delenv(k, raising=False)
    s = load_settings()
    assert s.planner_seeds and s.evidence_only_single_task and s.speculative_retrieval
    assert s.gemini_reasoning_effort == "minimal" and s.gemini_planner_reasoning_effort == "low"
    monkeypatch.setenv("EVIDENCE_ONLY_SINGLE_TASK", "0")
    assert load_settings().evidence_only_single_task is False


# ---------------------------------------------------------------- runtime vs evaluation requirements (option 10)


def _pins(path):
    import re

    out = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "-r", "-c")):
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]+\])?==([^\s;#]+)\s*$", line)
        assert m, f"{path.name}: not an exact pin: {line}"
        out[re.sub(r"[-_.]+", "-", m.group(1)).lower()] = m.group(3)
    return out


def test_eval_requirements_are_exact_pins_and_agree_with_the_runtime_lockfile(root):
    runtime, evals = _pins(root / "requirements.txt"), _pins(root / "requirements-eval.txt")
    assert {"ragas", "langchain-community", "datasets"} <= set(evals)
    assert not {"ragas", "langchain-community", "datasets", "pandas", "scipy"} & set(runtime)  # not in the image
    assert {k: v for k, v in evals.items() if k in runtime} == runtime  # one environment, same versions
    assert _pins(root / "requirements-eval.in")["ragas"] == evals["ragas"]
