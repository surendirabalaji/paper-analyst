"""Phase 4 live routing checks (REAL OpenAI + Gemini-via-OpenRouter calls; opt-in with RUN_LIVE=1).

1. Two ordinary questions run through the real graph, retriever and router. They must pick different
   synthesis providers with a rationale, and every routed call must return status ok from its provider.
2. A real fallback: the Gemini model id is made invalid, so OpenRouter rejects the planner call and the
   router falls back to OpenAI, traced as a separate `fallback` event.
Cost: about $0.06 per run at the prices in app/llm/policy.py.
"""

import asyncio
import dataclasses
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from phase2_fakes import drain  # noqa: E402

pytestmark = [pytest.mark.live, pytest.mark.anyio, pytest.mark.usefixtures("close_openai_clients")]  # P8b-D1


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
def real():
    from app.config import load_settings
    from app.llm.router import Router
    from app.rag import Retriever

    s = load_settings()
    if not (s.openai_api_key and s.openrouter_api_key):
        pytest.skip("OPENAI_API_KEY and OPENROUTER_API_KEY are required")
    return s, Retriever(s), Router(s)


def _ctx(real, query, router=None):
    from app.context import RequestContext
    from app.events import Emitter

    s, retriever, default_router = real
    return RequestContext(request_id=f"live-{abs(hash(query)) % 10_000}", query=query, k=s.top_k, settings=s,
                          router=router or default_router, retriever=retriever, emitter=Emitter("live"))


def ev(events, name):
    return [p for e, p in events if e == name]


async def _run(ctx):
    from app.graph import run_request

    await asyncio.wait_for(run_request(ctx), 150)
    return drain(ctx)


async def test_live_two_ordinary_questions_pick_different_providers(real):
    single = await _run(_ctx(real, "How does Self-RAG decide when to retrieve?"))
    compare = await _run(_ctx(real, "Compare how Reflexion and Self-Refine use feedback to improve outputs."))
    synth = {}
    for name, events in (("single", single), ("compare", compare)):
        done = ev(events, "done")[0]
        assert done["status"] in ("done", "partial"), done
        routes = ev(events, "llm_route")
        results = {r["decision_id"]: r for r in ev(events, "llm_result")}
        for d in routes:
            assert d["model_id"] and d["rationale"] and d["dimension"] and d["features"] is not None
            assert results[d["decision_id"]]["status"] == "ok", results[d["decision_id"]]
        syn = [d for d in routes if d["call_site"] == "synthesize"][0]
        synth[name] = syn
        assert results[syn["decision_id"]]["output_tokens"] > 0
        assert ev(events, "token") and ev(events, "citation")
    assert synth["single"]["provider"] == "gemini-openrouter" and synth["single"]["dimension"] == "cost"
    assert synth["compare"]["provider"] == "openai" and synth["compare"]["dimension"] == "capability"
    # the Gemini calls really went through OpenRouter to Google, and both providers answered
    gem = [r for r in ev(single, "llm_result") if r["provider"] == "gemini-openrouter"]
    assert gem and all(r["upstream_provider"] for r in gem)
    assert set(ev(compare, "done")[0]["providers_used"]) == {"gemini-openrouter", "openai"}


async def test_live_invalid_gemini_model_falls_back_to_openai(real):
    from langchain_core.messages import HumanMessage

    from app.llm.router import Router, call_llm

    s = dataclasses.replace(real[0], gemini_model="google/this-model-does-not-exist")
    ctx = _ctx(real, "fallback probe", router=Router(s))
    d = ctx.route("planner", "planner", {})
    assert d.provider == "gemini-openrouter"
    out = await call_llm(ctx, d, lambda c: c, [HumanMessage("Reply with the single word: ready")])
    events = drain(ctx)
    assert out.content
    fb = ev(events, "fallback")
    assert len(fb) == 1 and fb[0]["original_decision_id"] == d.decision_id and fb[0]["to_provider"] == "openai"
    statuses = [(r["provider"], r["status"]) for r in ev(events, "llm_result")]
    assert statuses[0] == ("gemini-openrouter", "error") and statuses[-1] == ("openai", "ok")
