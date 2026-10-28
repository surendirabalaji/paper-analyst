"""Tester Phase 8b checks (latency package B). MOCKED providers and retriever (no network, no spend).

Independent of the developer's tests/unit/test_phase8b_latency.py. Targets the edges of the new latency path:
cancellation during planning, discarded speculation never registering citation ids, evidence-only answers
that cite off-topic evidence, a corpus tool fault with a kept speculative search, tool-call bounds with
seeds, request isolation under concurrency, and a worker's free text never reaching synthesis in
evidence-only mode.
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage

from app.faults import Faults
from app.graph import run_request
from tests.phase2_fakes import FakeRetriever, Script, chunk, drain, make_ctx, tool_call

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


async def test_T8b_U1_cancel_during_planning_cancels_the_speculative_search():
    """Client disconnect while the planner runs: the speculative search must not outlive the request."""
    started = asyncio.Event()

    class SlowRetriever(FakeRetriever):
        async def search(self, query, k):
            self.calls.append((query, k))
            started.set()
            await asyncio.sleep(30)
            return []

    async def slow_plan(messages, kw):
        await asyncio.sleep(30)

    retr = SlowRetriever()
    ctx = make_ctx("How does alpha work?", Script(planner=slow_plan), retriever=retr, **ON)
    task = asyncio.create_task(run_request(ctx))
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    spec_task = ctx.speculative["task"]
    assert spec_task.done() and spec_task.cancelled()


async def test_T8b_U2_greeting_discards_speculation_and_registers_no_citations():
    s = Script(planner={"plan_summary": "greeting", "tasks": []}, synth_chunks=["Hello! Ask me about the papers."])
    retr = FakeRetriever()
    ctx = make_ctx("hello there", s, retriever=retr, **ON)
    final = await run_request(ctx)
    events = drain(ctx)
    spec = ev(events, "speculative_retrieval")
    assert len(spec) == 1 and spec[0]["status"] == "discarded"
    assert ev(events, "citation") == [] and ev(events, "retrieval") == []
    assert ctx.registry.get("C1") is None
    assert final["status"] == "done"


async def test_T8b_U3_evidence_only_answer_citing_off_topic_chunk_is_not_linked():
    """The model cites C1 although C1 was withheld (similarity 0.2): no citation link may be emitted for it."""
    low = FakeRetriever(lambda q: [chunk("2401.00001v1", "S1", 0, "Unrelated text.", score=0.2)])
    s = Script(planner={"plan_summary": "p", "tasks": [research_task()]}, worker=[AIMessage(content="DONE")],
               synth_chunks=["Canberra is the capital [C1]."])
    ctx = make_ctx("What is the capital of Australia?", s, retriever=low, **ON)
    final = await run_request(ctx)
    events = drain(ctx)
    assert [d["cite_id"] for d in ev(events, "citation")] == []
    done = ev(events, "done")[0]
    assert "C1" in (done.get("dropped_citations") or []) or "C1" in (final.get("excluded_evidence") or [])


async def test_T8b_U4_corpus_tool_fault_with_kept_speculation_fails_honestly():
    """tool:corpus_search:error must apply to the speculative seed too (it runs through _run_tool)."""
    s = Script(planner={"plan_summary": "p", "tasks": [research_task(["alpha"])]},
               worker=[AIMessage(content="DONE")], synth_chunks=["must not stream"])
    ctx = make_ctx("How does alpha work?", s, retriever=FakeRetriever(), **ON)
    ctx.faults = Faults.from_spec("tool:corpus_search:error", "header")
    final = await run_request(ctx)
    events = drain(ctx)
    results = ev(events, "tool_result")
    assert results and all(r["status"] != "ok" for r in results if r["tool"] == "corpus_search")
    wr = ev(events, "worker_result")[0]
    assert wr["status"] == "failed"
    assert ev(events, "citation") == []
    assert final["status"] == "failed"
    assert "must not stream" not in final["final_answer"]


async def test_T8b_U5_seeds_count_against_the_per_task_tool_bound():
    three = AIMessage(content="", tool_calls=[tool_call("corpus_search", {"query": f"q{i}"}, i) for i in range(3)])
    s = Script(planner={"plan_summary": "p", "tasks": [research_task(["alpha"])]}, worker=[three],
               synth_chunks=["ok [C1]"])
    retr = FakeRetriever()
    ctx = make_ctx("How does alpha work?", s, retriever=retr, **ON)
    await run_request(ctx)
    events = drain(ctx)
    calls = ev(events, "tool_call")
    assert sum(1 for c in calls if c.get("seeded_by")) == 2
    assert len(retr.calls) <= 4 and len([c for c in calls if c["tool"] == "corpus_search"]) <= 4


async def test_T8b_U6_concurrent_requests_keep_their_own_speculation_and_citations():
    def pick(q):
        doc = "2401.00001v1" if "alpha" in q else "2402.00002v1"
        return [chunk(doc, "S1", 0, f"text about {q}")]

    retr = FakeRetriever(pick)  # one shared retriever, as in production

    async def one(q, rid):
        s = Script(planner={"plan_summary": "p", "tasks": [research_task()]}, worker=[AIMessage(content="DONE")],
                   synth_chunks=["answer [C1]"])
        ctx = make_ctx(q, s, retriever=retr, request_id=rid, **ON)
        final = await run_request(ctx)
        return ctx, final, drain(ctx), s

    (ca, fa, ea, sa), (cb, fb, eb, sb) = await asyncio.gather(one("alpha question", "ra"), one("beta question", "rb"))
    for events, doc, other, script in ((ea, "2401.00001v1", "2402.00002v1", sa), (eb, "2402.00002v1", "2401.00001v1", sb)):
        retrieved = {c["doc_id"] for r in ev(events, "retrieval") for c in r["chunks"]}
        assert retrieved == {doc}
        assert other not in script.synth_messages[-1].content
        assert ev(events, "speculative_retrieval")[0]["status"] == "kept"
    assert ca.speculative["query"] == "alpha question" and cb.speculative["query"] == "beta question"


async def test_T8b_U7_evidence_only_worker_text_never_reaches_synthesis():
    """A worker that answers in prose (fabricated claim) instead of calling tools: in evidence-only mode its
    text is dropped; synthesis sees only the evidence-only note plus the real evidence."""
    s = Script(planner={"plan_summary": "p", "tasks": [research_task(["alpha"])]},
               worker=[AIMessage(content="FABRICATED-CLAIM: alpha was invented in 1850 [C9].")],
               synth_chunks=["ok [C1]"])
    ctx = make_ctx("How does alpha work?", s, retriever=FakeRetriever(), **ON)
    await run_request(ctx)
    human = s.synth_messages[-1].content
    assert "FABRICATED-CLAIM" not in human and "evidence-only task" in human


async def test_T8b_U8_switches_off_restore_findings_mode():
    """With all three switches off a single Research task writes findings again (rollback path works)."""
    s = Script(planner={"plan_summary": "p", "tasks": [research_task(["alpha"])]},
               worker=[AIMessage(content="", tool_calls=[tool_call("corpus_search", {"query": "alpha"})]),
                       AIMessage(content="- Alpha retrieves when uncertain [C1].")],
               synth_chunks=["ok [C1]"])
    ctx = make_ctx("How does alpha work?", s, retriever=FakeRetriever())  # dataclass defaults: off
    await run_request(ctx)
    events = drain(ctx)
    assert ev(events, "speculative_retrieval") == []
    assert not any(c.get("seeded_by") for c in ev(events, "tool_call"))
    assert not any(w.get("evidence_only") for w in ev(events, "worker_result"))
    assert "Alpha retrieves when uncertain" in s.synth_messages[-1].content
