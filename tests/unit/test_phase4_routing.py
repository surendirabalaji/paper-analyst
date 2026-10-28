"""Phase 4: routing policy, resilient calls (retry / fallback / cancel) and graph-level routing.

Policy tests are pure. Resilience tests use scripted LangChain chat models that raise real `openai`
exception types, so router.classify_error and the retry/fallback loop run unchanged (MOCKED providers).
"""

import asyncio
import pathlib
import sys

import httpx
import openai
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from phase2_fakes import FakeRouter, Script, chunk, drain, make_ctx, make_settings, plan, tool_call  # noqa: E402

from app.context import RequestContext  # noqa: E402
from app.events import Emitter  # noqa: E402
from app.graph import run_request  # noqa: E402
from app.llm import policy  # noqa: E402
from app.llm.policy import GEMINI, OPENAI  # noqa: E402
from app.llm.router import OpenRouterChat, Router, call_llm, stream_llm  # noqa: E402

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def ev(events, name):
    return [p for e, p in events if e == name]


# ---------------------------------------------------------------- policy (pure)


@pytest.mark.parametrize("site", ["planner", "worker_tool_loop", "worker_final", "synthesize_no_task"])
def test_structured_and_short_calls_go_to_the_cost_side(site):
    c = policy.choose(site, {})
    assert (c.provider, c.dimension) == (GEMINI, "cost")
    assert c.rationale


@pytest.mark.parametrize(
    "features, provider, dimension",
    [
        ({"n_tasks_ok": 1, "n_papers": 1, "evidence_tokens_est": 1500}, GEMINI, "cost"),
        ({"n_tasks_ok": 1, "n_papers": 2, "evidence_tokens_est": 2500}, GEMINI, "cost"),
        ({"n_tasks_ok": 1, "n_papers": 3, "evidence_tokens_est": 2500}, OPENAI, "capability"),
        ({"n_tasks_ok": 2, "n_papers": 1, "evidence_tokens_est": 800}, OPENAI, "capability"),
        ({"n_tasks_ok": 1, "n_papers": 2, "comparison": True}, OPENAI, "capability"),
        ({"n_tasks_ok": 1, "n_papers": 1, "comparison": True}, GEMINI, "cost"),
        ({"n_tasks_ok": 1, "n_papers": 1, "evidence_tokens_est": 5999}, GEMINI, "cost"),
        ({"n_tasks_ok": 1, "n_papers": 1, "evidence_tokens_est": 6000}, OPENAI, "capability"),
    ],
)
def test_synthesis_table_and_boundaries(features, provider, dimension):
    c = policy.choose("synthesize", features)
    assert (c.provider, c.dimension) == (provider, dimension)


def test_capability_rationale_names_the_observed_features():
    c = policy.choose("synthesize", {"n_tasks_ok": 2, "n_papers": 4, "comparison": True})
    assert "2 worker results" in c.rationale and "4 papers" in c.rationale and "comparison" in c.rationale


def test_availability_override_and_both_failed():
    c = policy.choose("planner", {}, failed={GEMINI: "timeout"})
    assert (c.provider, c.dimension) == (OPENAI, "availability") and "timeout" in c.rationale
    both = policy.choose("planner", {}, failed={GEMINI: "timeout", OPENAI: "rate_limit"})
    assert both.provider == GEMINI and both.dimension == "cost"  # nothing better; fallback layer reports it


def test_force_override_for_evaluation():
    c = policy.choose("synthesize", {"n_papers": 9}, force=GEMINI)
    assert (c.provider, c.dimension) == (GEMINI, "forced")
    assert policy.choose("planner", {}, force="bogus").dimension == "cost"


@pytest.mark.parametrize("q, cue", [
    ("Compare Reflexion and Self-Refine", True), ("ReAct vs. Toolformer", True), ("ReAct vs Toolformer", True),
    ("What is the difference between HyDE and RAG?", True), ("trade-offs of debate", True),
    ("How does Self-RAG decide when to retrieve?", False), ("What different memory types exist?", False),
    ("Why do multi-agent LLM systems fail?", False),
])
def test_comparison_cue(q, cue):
    assert policy.comparison_cue(q) is cue


def test_cost_estimate_uses_dated_price_table():
    assert policy.estimate_cost("gpt-6.1-sol", 1_000_000, 100_000) == pytest.approx(3.0)
    assert policy.estimate_cost("google/gemini-3.8-flash", 1_000_000, 0, cached_input_tokens=1_000_000) == pytest.approx(0.15)
    assert policy.estimate_cost("unknown-model", 10, 10) is None
    for p in policy.PRICES.values():
        assert p["checked"] and p["source"].startswith("https://")


def test_router_decision_carries_exact_model_ids_and_trace_fields():
    r = Router(make_settings(openai_model="gpt-6.1-sol", gemini_model="google/gemini-3.8-flash"))
    d = r.decide("synthesize", "synthesizer", {"n_tasks_ok": 2, "n_papers": 3})
    pub = d.public()
    assert pub["provider"] == OPENAI and pub["model_id"] == "gpt-6.1-sol" and pub["dimension"] == "capability"
    assert pub["features"]["n_papers"] == 3 and pub["price_checked"] == "2026-10-27" and pub["provider_label"]
    g = r.decide("planner", "planner", {})
    assert g.model_id == "google/gemini-3.8-flash" and g.reasoning_effort == "low" and g.timeout_s == 20.0


def test_openrouter_extras_upstream_provider_and_cost():
    assert OpenRouterChat._extra({"provider": "Google AI Studio", "usage": {"cost": 0,
                                  "cost_details": {"upstream_inference_cost": 0.0023}}}) == {
        "upstream_provider": "Google AI Studio", "reported_cost_usd": 0.0023}
    assert OpenRouterChat._extra({"usage": {"cost": 0.01}}) == {"reported_cost_usd": 0.01}
    assert OpenRouterChat._extra({}) == {}


# ---------------------------------------------------------------- resilient calls (mocked providers)


def _resp(code):
    return httpx.Response(code, request=httpx.Request("POST", "http://provider.invalid/v1/chat"))


def rate_limit():
    return openai.RateLimitError("rate limited", response=_resp(429), body=None)


def server_error():
    return openai.InternalServerError("upstream 503", response=_resp(503), body=None)


def auth_error():
    return openai.AuthenticationError("bad key sk-proj-SECRETSECRETSECRET99", response=_resp(401), body=None)


class Scripted(BaseChatModel):
    """Each call pops the next behaviour: an Exception to raise, a float to sleep, or a str reply."""

    behaviours: list
    stream_parts: list = []
    calls: int = 0

    @property
    def _llm_type(self):
        return "scripted"

    def _generate(self, *a, **k):
        raise RuntimeError("sync not expected")

    async def _next(self):
        self.calls += 1
        b = self.behaviours.pop(0) if self.behaviours else "ok"
        if isinstance(b, BaseException):
            raise b
        if isinstance(b, float):
            await asyncio.sleep(b)
            return "late"
        return b

    async def _agenerate(self, messages, stop=None, run_manager=None, **kw):
        text = await self._next()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content=text, usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}))])

    async def _astream(self, messages, stop=None, run_manager=None, **kw):
        for part in self.stream_parts:
            if isinstance(part, BaseException):
                raise part
            yield ChatGenerationChunk(message=AIMessageChunk(content=part))
        b = await self._next()
        yield ChatGenerationChunk(message=AIMessageChunk(content=b if b != "ok" else "",
                                  usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}))


class TwoProviderRouter(Router):
    def __init__(self, settings, models):
        super().__init__(settings)
        self.models = models

    def client(self, provider):
        return self.models[provider]


def ctx_with(models, **over):
    s = make_settings(**over)
    return RequestContext(request_id="r", query="q", k=6, settings=s, router=TwoProviderRouter(s, models),
                          retriever=None, emitter=Emitter("r"))


def assert_every_decision_has_a_result(events):
    ids = {d["decision_id"] for d in ev(events, "llm_route")} | {d["decision_id"] for d in ev(events, "fallback")}
    got = {d["decision_id"] for d in ev(events, "llm_result")}
    assert ids <= got, f"decisions without llm_result: {ids - got}"


async def test_transient_error_retries_once_then_falls_back_to_other_provider(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    g = Scripted(behaviours=[rate_limit(), server_error()])
    o = Scripted(behaviours=["from openai"])
    ctx = ctx_with({GEMINI: g, OPENAI: o})
    d = ctx.route("planner", "planner", {})
    out = await call_llm(ctx, d, lambda c: c, "hi")
    events = drain(ctx)
    assert out.content == "from openai" and g.calls == 2 and o.calls == 1
    assert len(ev(events, "llm_retry")) == 1
    fb = ev(events, "fallback")[0]
    assert fb["original_decision_id"] == d.decision_id and fb["decision_id"] != d.decision_id
    assert (fb["from_provider"], fb["to_provider"], fb["error_kind"]) == (GEMINI, OPENAI, "provider_unavailable")
    assert fb["dimension"] == "availability fallback"
    assert ev(events, "llm_route")[0]["provider"] == GEMINI  # original decision is not rewritten
    assert [r["status"] for r in ev(events, "llm_result")] == ["error", "error", "ok"]
    assert_every_decision_has_a_result(events)
    assert ctx.failed_providers == {GEMINI: "provider_unavailable"} and ctx.fallbacks == 1
    # later calls in the same request avoid the failed provider up front
    later = ctx.route("worker_tool_loop", "research", {})
    assert (later.provider, later.dimension) == (OPENAI, "availability")


async def _no_sleep(_):
    return None


async def test_auth_error_falls_back_without_retry_and_redacts():
    g = Scripted(behaviours=[auth_error()])
    o = Scripted(behaviours=["ok openai"])
    ctx = ctx_with({GEMINI: g, OPENAI: o})
    await call_llm(ctx, ctx.route("planner", "planner"), lambda c: c, "hi")
    events = drain(ctx)
    assert g.calls == 1 and not ev(events, "llm_retry")
    assert ev(events, "fallback")[0]["error_kind"] == "auth_or_config"
    assert "SECRETSECRETSECRET99" not in str(events)


async def test_both_providers_fail_raises_and_never_fakes_success(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    ctx = ctx_with({GEMINI: Scripted(behaviours=[auth_error()]), OPENAI: Scripted(behaviours=[auth_error()])})
    with pytest.raises(openai.AuthenticationError):
        await call_llm(ctx, ctx.route("planner", "planner"), lambda c: c, "hi")
    events = drain(ctx)
    assert all(r["status"] == "error" for r in ev(events, "llm_result"))
    assert len(ev(events, "fallback")) == 1  # only one fallback hop
    assert_every_decision_has_a_result(events)


async def test_fallback_can_be_disabled():
    ctx = ctx_with({GEMINI: Scripted(behaviours=[auth_error()]), OPENAI: Scripted(behaviours=["x"])},
                   fallback_enabled=False)
    with pytest.raises(openai.AuthenticationError):
        await call_llm(ctx, ctx.route("planner", "planner"), lambda c: c, "hi")
    assert not ev(drain(ctx), "fallback")


async def test_non_provider_error_propagates_without_fallback_but_has_result():
    ctx = ctx_with({GEMINI: Scripted(behaviours=[ValueError("bad parse")]), OPENAI: Scripted(behaviours=["x"])})
    with pytest.raises(ValueError):
        await call_llm(ctx, ctx.route("planner", "planner"), lambda c: c, "hi")
    events = drain(ctx)
    assert not ev(events, "fallback")
    assert ev(events, "llm_result")[0]["status"] == "error"


async def test_per_attempt_timeout_is_a_transient_error(monkeypatch):
    monkeypatch.setitem(policy.TIMEOUTS, "planner", 0.05)
    g = Scripted(behaviours=[1.0, 1.0])
    o = Scripted(behaviours=["openai ok"])
    ctx = ctx_with({GEMINI: g, OPENAI: o})
    out = await call_llm(ctx, ctx.route("planner", "planner"), lambda c: c, "hi")
    events = drain(ctx)
    assert out.content == "openai ok"
    assert [r["error_kind"] for r in ev(events, "llm_result")][:2] == ["timeout", "timeout"]


async def test_cancelled_call_still_emits_a_result():
    ctx = ctx_with({GEMINI: Scripted(behaviours=[5.0]), OPENAI: Scripted(behaviours=["x"])})
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(call_llm(ctx, ctx.route("planner", "planner"), lambda c: c, "hi"), 0.05)
    events = drain(ctx)
    assert [r["status"] for r in ev(events, "llm_result")] == ["cancelled"]
    assert_every_decision_has_a_result(events)


async def test_stream_falls_back_only_before_the_first_token(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    g = Scripted(behaviours=[], stream_parts=[server_error()])
    o = Scripted(behaviours=["ok"], stream_parts=["Hello ", "world"])
    ctx = ctx_with({GEMINI: g, OPENAI: o})
    seen = []
    text = await stream_llm(ctx, ctx.route("synthesize", "synthesizer", {"n_papers": 1}), lambda c: c, "hi", seen.append)
    events = drain(ctx)
    assert text == "Hello world" and seen == ["Hello ", "world"]
    assert len(ev(events, "fallback")) == 1
    ok = [r for r in ev(events, "llm_result") if r["status"] == "ok"][0]
    assert ok["provider"] == OPENAI and ok["stream_chunks"] == 2 and ok["ttft_ms"] is not None


async def test_stream_failure_after_tokens_is_raised_not_retried():
    g = Scripted(behaviours=[], stream_parts=["partial ", server_error()])
    o = Scripted(behaviours=["ok"], stream_parts=["never"])
    ctx = ctx_with({GEMINI: g, OPENAI: o})
    seen = []
    with pytest.raises(openai.InternalServerError):
        await stream_llm(ctx, ctx.route("synthesize", "synthesizer", {"n_papers": 1}), lambda c: c, "hi", seen.append)
    events = drain(ctx)
    assert seen == ["partial "] and o.calls == 0 and not ev(events, "fallback")
    assert ev(events, "llm_result")[0]["status"] == "error"


# ---------------------------------------------------------------- graph-level routing (mocked LLM + retriever)


def _rounds(text):
    return [AIMessage(content="", tool_calls=[tool_call("corpus_search", {"query": "alpha retrieval"})]), AIMessage(content=text)]


async def _run(ctx):
    await asyncio.wait_for(run_request(ctx), 20)
    return drain(ctx)


def _synth_route(events):
    return [d for d in ev(events, "llm_route") if d["call_site"].startswith("synthesize")][0]


async def test_graph_single_paper_question_synthesises_on_cost_side():
    from phase2_fakes import FakeRetriever
    s = Script(planner=plan(("research", "explain", [])), worker=_rounds("Alpha [C1]."), synth_chunks=["Alpha [C1]."])
    ctx = make_ctx("How does alpha decide when to retrieve?", s,
                   retriever=FakeRetriever(lambda q: [chunk("2401.00001v1", "S1", 0, "alpha text")]))
    events = await _run(ctx)
    routes = ev(events, "llm_route")
    assert {d["call_site"]: d["provider"] for d in routes}["planner"] == GEMINI
    syn = _synth_route(events)
    assert (syn["provider"], syn["dimension"]) == (GEMINI, "cost") and syn["features"]["n_papers"] == 1
    assert ev(events, "done")[0]["providers_used"] == [GEMINI]


async def test_graph_multi_paper_question_synthesises_on_capability_side():
    from phase2_fakes import FakeRetriever
    papers = [chunk(f"240{i}.0000{i}v1", "S1", 0, f"text {i}") for i in range(1, 4)]
    s = Script(planner=plan(("research", "survey", [])), worker=_rounds("A [C1]. B [C2]. C [C3]."),
               synth_chunks=["A [C1] B [C2] C [C3]"])
    ctx = make_ctx("Why do multi-agent systems fail?", s, retriever=FakeRetriever(lambda q: papers))
    events = await _run(ctx)
    syn = _synth_route(events)
    assert (syn["provider"], syn["dimension"]) == (OPENAI, "capability") and "3 papers" in syn["rationale"]
    assert ev(events, "done")[0]["providers_used"] == [GEMINI, OPENAI]


async def test_graph_worker_timeout_leaves_no_route_without_result():
    """P2-N4: the worker's in-flight LLM call is cut by the worker deadline; its route still gets a result.
    Since Phase 7 the attempt timeout is capped by the worker's deadline, so the call ends as a recorded
    `timeout` error rather than being cancelled from outside."""
    async def slow(messages, kw):
        await asyncio.sleep(5)

    s = Script(planner=plan(("research", "x", [])), worker=[slow, slow], synth_chunks=["none"])
    ctx = make_ctx("q", s, worker_timeout_s=0.2)
    events = await _run(ctx)
    assert_every_decision_has_a_result(events)
    cut = [r for r in ev(events, "llm_result") if r["agent"] == "research"]
    assert cut and all(r["status"] in ("error", "cancelled") for r in cut)
    assert "timeout" in [r["error_kind"] for r in cut]
    assert ev(events, "done")


# ---------------------------------------------------------------- P4-D1: cost accounting


def test_p4_d1_gemini_price_matches_openrouter_reported_byok_cost():
    # Observed on deployed calls (2026-10-27): 360 in / 112 out reported $0.00138; 9840 in / 364 out $0.01749.
    assert policy.estimate_cost("google/gemini-3.8-flash", 360, 112) == pytest.approx(0.00138, rel=1e-6)
    assert policy.estimate_cost("google/gemini-3.8-flash", 9840, 364) == pytest.approx(0.01749, rel=1e-6)


def _result_ctx():
    s = make_settings(gemini_model="google/gemini-3.8-flash", openai_model="gpt-6.1-sol")
    return RequestContext(request_id="r", query="q", k=6, settings=s, router=Router(s), retriever=None,
                          emitter=Emitter("r"))


def test_p4_d1_reported_cost_is_preferred_and_labelled():
    import time

    from app.llm.router import record_llm_result

    ctx = _result_ctx()
    d = ctx.route("planner", "planner")
    msg = AIMessage(content="x", usage_metadata={"input_tokens": 360, "output_tokens": 112, "total_tokens": 472},
                    response_metadata={"reported_cost_usd": 0.00138, "upstream_provider": "Google AI Studio"})
    record_llm_result(ctx, d, msg, time.perf_counter())
    r = ev(drain(ctx), "llm_result")[0]
    assert (r["cost_usd"], r["cost_source"]) == (0.00138, "reported")
    assert r["est_cost_usd"] == pytest.approx(r["reported_cost_usd"], rel=1e-6)
    assert ctx.tokens_by_provider[GEMINI]["cost_usd"] == 0.00138 and ctx.cost_basis == {"reported": 1}


def test_p4_d1_estimate_used_when_provider_reports_no_cost():
    import time

    from app.llm.router import record_llm_result

    ctx = _result_ctx()
    d = ctx.router.decide("synthesize", "synthesizer", {"n_papers": 3})
    msg = AIMessage(content="x", usage_metadata={"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100})
    record_llm_result(ctx, d, msg, time.perf_counter())
    r = ev(drain(ctx), "llm_result")[0]
    assert r["provider"] == OPENAI and r["cost_source"] == "estimated" and r["cost_usd"] == pytest.approx(0.003)
    assert ctx.cost_basis == {"estimated": 1}
