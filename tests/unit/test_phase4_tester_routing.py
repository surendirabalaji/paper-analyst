"""Tester-owned Phase 4 checks (MOCKED providers; no network, no paid calls).

Complements tests/unit/test_phase4_routing.py with: exact threshold boundaries, agent-independence of
the policy (no per-worker hard-coding), held-out comparison phrasings, cancellation during a fallback
attempt and during retry backoff, redaction of OpenRouter-style keys in llm_result errors, missing
features, and the request-budget interaction of synthesis retries (documents P4-N1).
"""

import asyncio
import pathlib
import sys
import time

import httpx
import openai
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from phase2_fakes import drain, make_settings  # noqa: E402

from app.context import RequestContext  # noqa: E402
from app.events import Emitter, safe_error  # noqa: E402
from app.llm import policy  # noqa: E402
from app.llm.policy import GEMINI, OPENAI  # noqa: E402
from app.llm.router import call_llm, stream_llm  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_phase4_routing import Scripted, TwoProviderRouter  # noqa: E402

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def ev(events, name):
    return [p for e, p in events if e == name]


def ctx_with(models, **over):
    s = make_settings(**over)
    return RequestContext(request_id="t", query="q", k=6, settings=s, router=TwoProviderRouter(s, models),
                          retriever=None, emitter=Emitter("t"))


def every_decision_has_result(events):
    ids = {d["decision_id"] for d in ev(events, "llm_route")} | {d["decision_id"] for d in ev(events, "fallback")}
    return ids - {d["decision_id"] for d in ev(events, "llm_result")}


def _resp(code):
    return httpx.Response(code, request=httpx.Request("POST", "https://example.invalid/v1/chat"))


# ---------------------------------------------------------------- policy boundaries (pure)

BASE = {"n_tasks_ok": 1, "n_papers": 1, "comparison": False, "evidence_tokens_est": 100}


@pytest.mark.parametrize("over,provider", [
    ({}, GEMINI),
    ({"n_tasks_ok": 2}, OPENAI),                     # at threshold
    ({"n_papers": 2}, GEMINI),                       # one below paper threshold, no comparison
    ({"n_papers": 3}, OPENAI),                       # at paper threshold
    ({"comparison": True, "n_papers": 1}, GEMINI),   # comparison cue but evidence from one paper only
    ({"comparison": True, "n_papers": 2}, OPENAI),
    ({"evidence_tokens_est": 5999}, GEMINI),
    ({"evidence_tokens_est": 6000}, OPENAI),
])
def test_tc_p4_01_synthesis_threshold_boundaries(over, provider):
    c = policy.choose("synthesize", {**BASE, **over})
    assert c.provider == provider
    assert c.dimension == ("capability" if provider == OPENAI else "cost")
    assert c.rationale


def test_tc_p4_02_missing_features_default_to_cost_side_without_crash():
    c = policy.choose("synthesize", {})
    assert (c.provider, c.dimension) == (GEMINI, "cost")
    assert "0 papers" in c.rationale


def test_tc_p4_03_policy_ignores_agent_identity():
    """Routing depends on call site + features, not on which worker calls (no fixed provider per worker)."""
    from app.config import Settings  # noqa: F401
    from app.llm.router import Router

    r = Router(make_settings(openai_model="m-o", gemini_model="m-g"))
    for site, feats in [("synthesize", {"n_papers": 4}), ("synthesize", {"n_papers": 1}),
                        ("worker_tool_loop", {"round": 1})]:
        provs = {r.decide(site, agent, feats).provider for agent in ("research", "summary", "data", "anything")}
        assert len(provs) == 1, (site, provs)


@pytest.mark.parametrize("q,cue", [
    ("How do ReAct and Toolformer differ in how they call tools?", True),        # held-out paraphrase
    ("Which is better for long-horizon tasks, tree search or reflection?", True),
    ("What are the different components of the Voyager agent?", False),         # 'different' is not a cue
    ("Contrast the memory designs in MemGPT and Generative Agents", True),
    ("Summarise the evaluation setup of SWE-agent", False),
    ("How does AutoGen stack up against CAMEL for role-play?", False),          # known miss: no lexical cue
])
def test_tc_p4_04_comparison_cue_heldout_phrasings(q, cue):
    assert policy.comparison_cue(q) is cue


def test_tc_p4_05_force_override_not_in_deploy_allowlist():
    src = (pathlib.Path(__file__).resolve().parents[2] / "scripts" / "env_to_yaml.py").read_text()
    assert "ROUTER_FORCE_PROVIDER" not in [ln.strip().strip("\",'") for ln in src.splitlines()
                                           if ln.strip().startswith(("\"", "'"))]


# ---------------------------------------------------------------- resilience (mocked providers)


async def _no_sleep(_):
    return None


async def test_tc_p4_06_cancel_during_fallback_attempt_records_cancelled_result():
    auth = openai.AuthenticationError("bad", response=_resp(401), body=None)
    ctx = ctx_with({GEMINI: Scripted(behaviours=[auth]), OPENAI: Scripted(behaviours=[5.0])})
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(call_llm(ctx, ctx.route("planner", "planner"), lambda c: c, "hi"), 0.2)
    events = drain(ctx)
    fb = ev(events, "fallback")
    assert len(fb) == 1
    statuses = [(r["decision_id"] == fb[0]["decision_id"], r["status"]) for r in ev(events, "llm_result")]
    assert statuses == [(False, "error"), (True, "cancelled")]
    assert not every_decision_has_result(events)


async def test_tc_p4_07_cancel_during_retry_backoff_leaves_no_orphan_route():
    rl = openai.RateLimitError("429", response=_resp(429), body=None)
    ctx = ctx_with({GEMINI: Scripted(behaviours=[rl, "never"]), OPENAI: Scripted(behaviours=["x"])})
    # real asyncio.sleep: backoff is >= 0.35 s, cancel at 0.1 s while sleeping
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(call_llm(ctx, ctx.route("planner", "planner"), lambda c: c, "hi"), 0.1)
    events = drain(ctx)
    assert len(ev(events, "llm_retry")) == 1
    assert not every_decision_has_result(events)


async def test_tc_p4_08_openrouter_key_canary_redacted_in_llm_result_and_fallback(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    canary = "sk-or-v1-CANARY0123456789abcdefCANARY"
    err = openai.AuthenticationError(f"No auth credentials found for key {canary} at "
                                     f"https://mcp.tavily.com/mcp/?tavilyApiKey=tvly-CANARYCANARY",
                                     response=_resp(401), body=None)
    ctx = ctx_with({GEMINI: Scripted(behaviours=[err]), OPENAI: Scripted(behaviours=["fine"])})
    await call_llm(ctx, ctx.route("planner", "planner"), lambda c: c, "hi")
    blob = str(drain(ctx))
    assert "CANARY0123456789" not in blob and "tvly-CANARYCANARY" not in blob
    assert "REDACTED" in blob


def test_tc_p4_09_safe_error_redacts_bearer_header():
    s = safe_error(RuntimeError("401 Authorization: Bearer abc.def.ghi rejected"))
    assert "abc.def.ghi" not in s


async def test_tc_p4_10_max_attempts_per_call_is_four(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    mk = lambda: openai.InternalServerError("500", response=_resp(500), body=None)  # noqa: E731
    g, o = Scripted(behaviours=[mk() for _ in range(9)]), Scripted(behaviours=[mk() for _ in range(9)])
    ctx = ctx_with({GEMINI: g, OPENAI: o})
    with pytest.raises(openai.InternalServerError):
        await call_llm(ctx, ctx.route("planner", "planner"), lambda c: c, "hi")
    assert (g.calls, o.calls) == (2, 2)
    events = drain(ctx)
    assert len(ev(events, "llm_result")) == 4 and len(ev(events, "fallback")) == 1


async def test_tc_p4_11_synthesis_retry_can_overrun_request_budget(monkeypatch):
    """Documents P4-N1 (non-blocking, Phase 7 scope): per-attempt timeouts are not clamped to the
    remaining request budget, so a slow synthesis plus its retry can outlive REQUEST_BUDGET_S.
    Scaled down: budget 12.5 s, synthesis attempt timeout 7 s (> the 5 s retry guard), provider never answers."""
    g = Scripted(behaviours=[30.0, 30.0], stream_parts=[])
    o = Scripted(behaviours=[30.0], stream_parts=[])
    ctx = ctx_with({GEMINI: g, OPENAI: o}, request_budget_s=12.5, llm_timeout_s=7.0)
    t0 = time.perf_counter()
    with pytest.raises(BaseException):
        await stream_llm(ctx, ctx.route("synthesize", "synthesizer", {"n_papers": 1}), lambda c: c, "hi",
                         lambda d: None)
    wall = time.perf_counter() - t0
    # bounded (no runaway), but it overruns the 12.5 s request budget (attempt 2 starts with 5.5 s left)
    assert wall < 2 * 7.0 + 1
    assert wall > ctx.settings.request_budget_s
    assert not every_decision_has_result(drain(ctx))


# ---------------------------------------------------------------- P4-D1 retest (cost basis), MOCKED payloads
# Payload shapes copied from a real OpenRouter BYOK response captured 2026-10-27
# (eval/results/testing/phase4-retest-d1-20261027T1645Z/raw_openrouter_usage.json): cost=0, is_byok=true,
# cost_details.upstream_inference_cost = prompt*1.50e-6 + completion*7.50e-6.

from langchain_core.messages import AIMessage  # noqa: E402

from app.llm.router import OpenRouterChat, record_llm_result  # noqa: E402

GM, OM = "google/gemini-3.8-flash", "gpt-6.1-sol"


def _usage(cost, upstream, p=8, c=1):
    u = {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c, "cost": cost, "is_byok": True}
    if upstream is not None:
        u["cost_details"] = {"upstream_inference_cost": upstream}
    return {"provider": "Google AI Studio", "usage": u}


def test_tc_p4_12_byok_zero_cost_uses_upstream_and_matches_price_table():
    extra = OpenRouterChat._extra(_usage(0, 1.95e-05))
    assert extra["reported_cost_usd"] == 1.95e-05
    # estimate_cost rounds to 6 decimals (1e-6 USD), so compare within that rounding unit.
    assert policy.estimate_cost(GM, 8, 1) == pytest.approx(extra["reported_cost_usd"], abs=1e-6)
    assert policy.estimate_cost(GM, 357, 77) == pytest.approx(0.001113, abs=1e-9)  # deployed planner call


def test_tc_p4_13_non_byok_cost_is_used_and_absent_cost_gives_no_reported_field():
    assert OpenRouterChat._extra(_usage(0.0042, None))["reported_cost_usd"] == 0.0042
    assert "reported_cost_usd" not in OpenRouterChat._extra(_usage(0, None))


@pytest.mark.xfail(strict=True, reason="P4-N6 (latent, Low): if OpenRouter charges a BYOK fee (cost>0) the "
                   "upstream Google charge is dropped; total spend would be cost + upstream_inference_cost")
def test_tc_p4_14_byok_fee_plus_upstream_should_both_be_counted():
    extra = OpenRouterChat._extra(_usage(0.000001, 1.95e-05))
    assert extra["reported_cost_usd"] == pytest.approx(0.000001 + 1.95e-05)


def _rctx(gm=GM):
    from app.llm.router import Router
    st = make_settings(gemini_model=gm, openai_model=OM)
    return RequestContext(request_id="t", query="q", k=6, settings=st, router=Router(st), retriever=None,
                          emitter=Emitter("t"))


def test_tc_p4_15_gemini_without_reported_cost_falls_back_to_dated_estimate():
    ctx = _rctx()
    d = ctx.route("planner", "planner")
    assert d.provider == GEMINI
    msg = AIMessage(content="x", usage_metadata={"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100})
    record_llm_result(ctx, d, msg, time.perf_counter())
    r = ev(drain(ctx), "llm_result")[0]
    assert r["cost_source"] == "estimated" and r["reported_cost_usd"] is None
    assert r["cost_usd"] == pytest.approx((1000 * 1.50 + 100 * 7.50) / 1e6)
    assert ctx.cost_basis == {"estimated": 1}


def test_tc_p4_16_unknown_model_and_failed_call_are_unpriced_not_zero():
    ctx = _rctx("google/some-new-model")
    d = ctx.route("planner", "planner")
    msg = AIMessage(content="x", usage_metadata={"input_tokens": 10, "output_tokens": 1, "total_tokens": 11})
    record_llm_result(ctx, d, msg, time.perf_counter())
    record_llm_result(ctx, d, None, time.perf_counter(), status="error", error_kind="timeout")
    a, b = ev(drain(ctx), "llm_result")
    assert a["cost_usd"] is None and a["cost_source"] is None and a["est_cost_usd"] is None
    assert b["cost_usd"] is None and b["cost_source"] is None
    assert ctx.cost_basis == {}
