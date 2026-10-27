"""Tester Phase 4 live fault checks (REAL OpenAI + Gemini-via-OpenRouter; opt-in with RUN_LIVE=1).

F1  Real 401: OpenRouter key replaced by a synthetic canary. The full graph runs; the first Gemini call
    gets a genuine 401 from OpenRouter, falls back to OpenAI as a separate `fallback` event, and later
    calls in the request route around Gemini with dimension "availability". Canary never appears.
F2  Real timeout: the original planner decision's per-attempt timeout is set to 50 ms against real Gemini: two timeouts
    (one llm_retry), then a real OpenAI fallback that succeeds.
F3  Real cancellation: an OpenAI synthesis stream is cancelled after its first delta; the attempt still
    gets an llm_result with status "cancelled".
Cost: about $0.04 per run. No credential values are printed.
"""

import asyncio
import dataclasses
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from phase2_fakes import drain  # noqa: E402

pytestmark = [pytest.mark.live, pytest.mark.anyio]
CANARY = "sk-or-v1-TESTERCANARY000000000000000000000000000000000000000000000000"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
def real():
    from app.config import load_settings
    from app.rag import Retriever

    s = load_settings()
    if not (s.openai_api_key and s.openrouter_api_key):
        pytest.skip("OPENAI_API_KEY and OPENROUTER_API_KEY are required")
    return s, Retriever(s)


def _ctx(s, retriever, query, rid):
    from app.context import RequestContext
    from app.events import Emitter
    from app.llm.router import Router

    return RequestContext(request_id=rid, query=query, k=s.top_k, settings=s, router=Router(s),
                          retriever=retriever, emitter=Emitter(rid))


def ev(events, name):
    return [p for e, p in events if e == name]


def orphans(events):
    ids = {d["decision_id"] for d in ev(events, "llm_route")} | {d["decision_id"] for d in ev(events, "fallback")}
    return ids - {d["decision_id"] for d in ev(events, "llm_result")}


async def test_f1_real_401_falls_back_and_later_calls_route_by_availability(real):
    from app.graph import run_request

    s, retriever = real
    bad = dataclasses.replace(s, openrouter_api_key=CANARY)
    ctx = _ctx(bad, retriever, "How does SWE-agent's agent-computer interface help with editing files?", "f1")
    await asyncio.wait_for(run_request(ctx), 150)
    events = drain(ctx)
    blob = str(events)
    assert "TESTERCANARY" not in blob
    fb = ev(events, "fallback")
    assert len(fb) == 1 and fb[0]["from_provider"] == "gemini-openrouter" and fb[0]["to_provider"] == "openai"
    assert fb[0]["error_kind"] == "auth_or_config" and fb[0]["call_site"] == "planner"
    first = ev(events, "llm_result")[0]
    assert first["status"] == "error" and first["provider"] == "gemini-openrouter"
    later = [d for d in ev(events, "llm_route") if d["call_site"] != "planner"]
    assert later and all(d["provider"] == "openai" for d in later)
    assert any(d["dimension"] == "availability" for d in later)
    done = ev(events, "done")[0]
    assert done["status"] in ("done", "partial") and done["fallbacks"] == 1
    assert done["providers_used"] == ["openai"]  # Gemini had calls but only errors
    assert not orphans(events)
    assert ev(events, "token")


async def test_f2_real_timeout_retries_then_falls_back(real):
    from app.llm.router import call_llm
    from langchain_core.messages import HumanMessage

    s, retriever = real
    ctx = _ctx(s, retriever, "q", "f2")
    d = ctx.route("planner", "planner", {"query_chars": 1})
    d.timeout_s = 0.05  # only the original (Gemini) decision; the fallback decision keeps the real 20 s
    out = await call_llm(ctx, d, lambda c: c, [HumanMessage("Reply with the single word: ok")])
    events = drain(ctx)
    assert out.content
    res = ev(events, "llm_result")
    print([{k: r.get(k) for k in ("provider", "attempt", "status", "error_kind", "error", "latency_ms")} for r in res])
    # attempt 1 may instead hit a stale pooled connection left by an earlier test's event loop (harness
    # artifact, see F3); attempt 2 is the real 50 ms timeout
    assert [(r["provider"], r["status"]) for r in res[:2]] == [("gemini-openrouter", "error")] * 2
    assert res[0]["error_kind"] in ("timeout", "provider_unavailable") and res[1]["error_kind"] == "timeout"
    assert res[2]["provider"] == "openai" and res[2]["status"] == "ok" and res[2]["model_returned"]
    assert len(ev(events, "llm_retry")) == 1 and len(ev(events, "fallback")) == 1
    assert not orphans(events)


async def test_f3_real_stream_cancel_records_cancelled(real):
    from app.llm.router import stream_llm
    from langchain_core.messages import HumanMessage

    s, retriever = real
    ctx = _ctx(s, retriever, "q", "f3")
    d = ctx.route("synthesize", "synthesizer", {"n_papers": 5})
    assert d.provider == "openai"
    got = asyncio.Event()

    async def run():
        await stream_llm(ctx, d, lambda c: c, [HumanMessage("Count slowly from 1 to 200, one number per line.")],
                         lambda _delta: got.set())

    t = asyncio.create_task(run())
    await asyncio.wait_for(got.wait(), 30)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    events = drain(ctx)
    res = ev(events, "llm_result")
    detail = [{k: r.get(k) for k in ("attempt", "status", "error_kind", "error", "ttft_ms", "latency_ms")} for r in res]
    # Under pytest each test has its own event loop while langchain-openai caches a pooled httpx client,
    # so a stale pooled connection may fail the first attempt (connection error, retried by the router).
    # That harness artifact is allowed; the cancelled attempt must still be recorded.
    assert res[-1]["status"] == "cancelled" and res[-1]["ttft_ms"] is not None, detail
    assert all(r["status"] == "error" and r["error_kind"] == "provider_unavailable" for r in res[:-1]), detail
    assert not orphans(events)
