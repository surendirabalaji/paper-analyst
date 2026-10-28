"""Generation-provider router and resilient, traced LLM calls.

Every LLM call goes through `Router.decide()` (the hybrid policy in policy.py) and then through
`call_llm()` or `stream_llm()`, which add:

- a per-attempt timeout (policy.TIMEOUTS, else LLM_TIMEOUT_S), capped by the time left in the request
  budget or the running worker's deadline (`ctx.time_left()`, P4-N1); SDK retries are off so retries are ours;
- one retry with jittered backoff on transient errors (timeout, 429, 5xx, connection);
- then a fallback to the other provider (directly for 401/403/404/400), emitted as a separate
  `fallback` event that names the original decision; a stream falls back only before its first token;
- an `llm_result` event for every attempt, with status ok / error / cancelled, so a failed or
  cancelled call never leaves its route without a result;
- an empty completion (no text, no tool calls, no output tokens; streamed or not) is `empty_response`,
  which is retried and then falls back like any transient error (P5-N1);
- fault injection hooks (app/faults.py, `provider:<name>:<mode>`) before each attempt; injected failures
  are marked `injected: true` in llm_result.

Trace events: `llm_route` (decision: agent, call site, provider, exact model id, decision inputs,
dimension, rationale), `llm_retry`, `fallback`, `llm_result` (tokens, reasoning tokens, latency, TTFT,
OpenRouter upstream provider, estimated and provider-reported cost).
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable

import openai
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.events import safe_error
from app.faults import InjectedFault, is_injected, maybe_fail
from app.llm import policy
from app.llm.policy import GEMINI, OPENAI

PROVIDER_LABELS = {
    OPENAI: "OpenAI (direct API)",
    GEMINI: "Google Gemini (served through OpenRouter)",
}
TRANSIENT = {"timeout", "rate_limit", "provider_unavailable", "empty_response"}
MAX_TRIES_PER_PROVIDER = 2
MIN_ATTEMPT_S = 0.25  # below this much time left, an attempt is not started (budget_exhausted)
MIN_RETRY_S = 5.0  # a retry or fallback needs at least this much time left


class OpenRouterChat(ChatOpenAI):
    """ChatOpenAI against OpenRouter that keeps two fields LangChain drops: the upstream provider that
    served the call (e.g. "Google AI Studio") and OpenRouter's reported cost. They are attached once per
    response (on the usage chunk when streaming) because LangChain concatenates string metadata across
    stream chunks."""

    @staticmethod
    def _extra(d: dict) -> dict:
        out = {}
        if d.get("provider"):
            out["upstream_provider"] = d["provider"]
        usage = d.get("usage") or {}
        cost = usage.get("cost") or 0
        upstream = (usage.get("cost_details") or {}).get("upstream_inference_cost") or 0
        if usage.get("is_byok"):
            # BYOK: OpenRouter's own fee (usually 0) plus what the upstream provider charges (P4-N6).
            total = cost + upstream
        else:
            total = cost or upstream
        if total:
            out["reported_cost_usd"] = round(total, 10)
        return out

    def _convert_chunk_to_generation_chunk(self, chunk, default_chunk_class, base_generation_info):
        g = super()._convert_chunk_to_generation_chunk(chunk, default_chunk_class, base_generation_info)
        if g is not None and chunk.get("usage"):
            g.message.response_metadata.update(self._extra(chunk))
        return g

    def _create_chat_result(self, response, generation_info=None):
        r = super()._create_chat_result(response, generation_info)
        d = response if isinstance(response, dict) else response.model_dump()
        for g in r.generations:
            g.message.response_metadata.update(self._extra(d))
        return r


@dataclass
class RouteDecision:
    call_site: str
    agent: str
    provider: str
    model_id: str
    dimension: str
    rationale: str
    features: dict[str, Any] = field(default_factory=dict)
    reasoning_effort: str | None = None
    timeout_s: float = 60.0
    price_checked: str | None = None
    decision_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d["provider_label"] = PROVIDER_LABELS.get(self.provider, self.provider)
        return d


class ProviderCallError(Exception):
    """An LLM attempt failed with a provider-side error; `kind` drives retry/fallback."""

    def __init__(self, kind: str, original: BaseException):
        super().__init__(str(original))
        self.kind = kind
        self.original = original


def text_of(content: Any) -> str:
    """Message content as plain text (content may be a str or a list of content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "")


def usage_of(msg: BaseMessage | None) -> dict[str, int]:
    u = getattr(msg, "usage_metadata", None) or {}
    out_d = u.get("output_token_details") or {}
    in_d = u.get("input_token_details") or {}
    return {
        "input_tokens": int(u.get("input_tokens", 0)),
        "output_tokens": int(u.get("output_tokens", 0)),
        "reasoning_tokens": int(out_d.get("reasoning") or 0),
        "cached_input_tokens": int(in_d.get("cache_read") or 0),
    }


def _model_returned(meta: dict) -> str | None:
    name = meta.get("model_name")
    if isinstance(name, str) and len(name) % 2 == 0 and name[: len(name) // 2] * 2 == name:
        return name[: len(name) // 2]  # merged stream chunks repeat the name once per carrying chunk
    return name


class BudgetExhausted(Exception):
    """No time left in the request budget (or the worker's deadline) to start another LLM attempt."""

    kind = "budget_exhausted"


def classify_error(e: BaseException) -> str:
    if isinstance(e, (ProviderCallError, InjectedFault, BudgetExhausted)):
        return e.kind
    if isinstance(e, (asyncio.TimeoutError, TimeoutError, openai.APITimeoutError)):
        return "timeout"
    if isinstance(e, openai.RateLimitError):
        return "rate_limit"
    if isinstance(e, (openai.APIConnectionError, openai.InternalServerError)):
        return "provider_unavailable"
    if isinstance(e, (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError)):
        return "auth_or_config"
    if isinstance(e, openai.BadRequestError):
        return "bad_request"
    if isinstance(e, openai.APIStatusError):
        return "provider_unavailable" if getattr(e, "status_code", 0) >= 500 else "provider_error"
    return "worker_error"


class EmptyResponse(Exception):
    """The provider returned HTTP 200 with no text, no tool calls and no usage (seen from Gemini via
    OpenRouter on Cloud Run, 2026-10-27). Treated as a transient provider failure: retry, then fall back."""


def is_empty_completion(msg: Any) -> bool:
    if not isinstance(msg, AIMessage):
        return False
    usage = msg.usage_metadata or {}
    return not text_of(msg.content).strip() and not msg.tool_calls and not usage.get("output_tokens")


def is_provider_error(e: BaseException) -> bool:
    return isinstance(e, (asyncio.TimeoutError, TimeoutError, openai.APIError, InjectedFault, BudgetExhausted))


def attempt_timeout(ctx, dec: RouteDecision) -> float:
    """The call site's timeout, capped by the time left in this request/stage (P2-N7, P4-N1)."""
    left = ctx.time_left()
    if left < MIN_ATTEMPT_S:
        raise BudgetExhausted(f"{left:.1f} s left in the request budget; not starting another LLM attempt")
    return min(dec.timeout_s, left)


class Router:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._clients: dict[str, ChatOpenAI] = {}

    def model_id(self, provider: str) -> str:
        return self.settings.openai_model if provider == OPENAI else self.settings.gemini_model

    def reasoning_effort(self, provider: str, call_site: str = "") -> str | None:
        """Gemini runs at `minimal` effort except the planner, which keeps `low` (plan quality, Phase 8b)."""
        s = self.settings
        if provider == OPENAI:
            return s.openai_reasoning_effort or None
        if call_site == "planner" and s.gemini_planner_reasoning_effort:
            return s.gemini_planner_reasoning_effort
        return s.gemini_reasoning_effort or None

    def client_for(self, dec: "RouteDecision") -> ChatOpenAI:
        """The client for one routed call: the provider's default client, or one with the call site's own
        reasoning effort (the planner's, Phase 8b)."""
        if dec.reasoning_effort == self.reasoning_effort(dec.provider):
            return self.client(dec.provider)
        return self._build(dec.provider, dec.reasoning_effort)

    def client(self, provider: str) -> ChatOpenAI:
        return self._build(provider, self.reasoning_effort(provider))

    def _build(self, provider: str, effort: str | None) -> ChatOpenAI:
        """One cached client per (provider, reasoning effort)."""
        key = f"{provider}:{effort or ''}"
        if key not in self._clients:
            s = self.settings
            common = dict(stream_usage=True, timeout=s.llm_timeout_s, max_retries=0)  # retries are ours
            if provider == OPENAI:
                kw = {"reasoning_effort": effort} if effort else {}
                self._clients[key] = ChatOpenAI(model=s.openai_model, api_key=s.openai_api_key, **common, **kw)
            elif provider == GEMINI:
                body: dict[str, Any] = {"usage": {"include": True}}
                if effort:  # Gemini 3.8 cannot disable reasoning; low/minimal effort keeps it near zero
                    body["reasoning"] = {"effort": effort}
                self._clients[key] = OpenRouterChat(
                    model=s.gemini_model, base_url=s.openrouter_base_url, api_key=s.openrouter_api_key,
                    extra_body=body, **common,
                )
            else:
                raise ValueError(f"unknown provider {provider!r}")
        return self._clients[key]

    def _decision(self, call_site: str, agent: str, choice: policy.Choice, features: dict) -> RouteDecision:
        model = self.model_id(choice.provider)
        return RouteDecision(
            call_site=call_site,
            agent=agent,
            provider=choice.provider,
            model_id=model,
            dimension=choice.dimension,
            rationale=choice.rationale,
            features=features,
            reasoning_effort=self.reasoning_effort(choice.provider, call_site),
            timeout_s=policy.TIMEOUTS.get(call_site, self.settings.llm_timeout_s),
            price_checked=(policy.PRICES.get(model) or {}).get("checked"),
        )

    def decide(self, call_site: str, agent: str, features: dict[str, Any] | None = None,
               failed: dict[str, str] | None = None) -> RouteDecision:
        features = dict(features or {})
        choice = policy.choose(call_site, features, failed=failed, force=self.settings.router_force_provider)
        return self._decision(call_site, agent, choice, features)

    def fallback_decision(self, original: RouteDecision, error_kind: str) -> RouteDecision:
        to = policy.other(original.provider)
        choice = policy.Choice(to, "availability fallback",
                               f"{original.provider} failed ({error_kind}); retrying this call on {to}.")
        return self._decision(original.call_site, original.agent, choice, original.features)


# ---------------------------------------------------------------- traced calls


def record_llm_result(ctx, decision: RouteDecision, msg: BaseMessage | None, t0: float, *, status: str = "ok",
                      attempt: int = 1, ttft_ms: int | None = None, error_kind: str | None = None,
                      error: str | None = None, stream_stats: dict | None = None, injected: bool = False) -> None:
    latency_ms = int((time.perf_counter() - t0) * 1000)
    usage = usage_of(msg)
    meta = (getattr(msg, "response_metadata", None) or {}) if msg is not None else {}
    est = policy.estimate_cost(decision.model_id, usage["input_tokens"], usage["output_tokens"],
                               usage["cached_input_tokens"]) if msg is not None else None
    # Prefer what the provider reports (OpenRouter returns a per-call cost); otherwise our dated estimate.
    reported = meta.get("reported_cost_usd")
    cost, cost_source = (reported, "reported") if reported is not None else (est, "estimated" if est is not None else None)
    ctx.llm_ms_sum += latency_ms
    tb = ctx.tokens_by_provider.setdefault(
        decision.provider, {"input_tokens": 0, "output_tokens": 0, "calls": 0, "errors": 0, "cost_usd": 0.0})
    tb["input_tokens"] += usage["input_tokens"]
    tb["output_tokens"] += usage["output_tokens"]
    tb["calls"] += 1
    if status != "ok":
        tb["errors"] += 1
    if cost:
        tb["cost_usd"] = round(tb["cost_usd"] + cost, 6)
        ctx.cost_basis[cost_source] = ctx.cost_basis.get(cost_source, 0) + 1
    ctx.emitter.emit(
        "llm_result",
        decision_id=decision.decision_id,
        agent=decision.agent,
        call_site=decision.call_site,
        provider=decision.provider,
        model_id=decision.model_id,
        model_returned=_model_returned(meta),
        upstream_provider=meta.get("upstream_provider"),
        status=status,
        attempt=attempt,
        error_kind=error_kind,
        error=error,
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        cost_usd=cost,
        cost_source=cost_source,
        est_cost_usd=est,
        reported_cost_usd=reported,
        **usage,
        **(stream_stats or {}),
        **({"injected": True} if injected else {}),
    )


async def _resilient(ctx, decision: RouteDecision, attempt_fn: Callable[[RouteDecision, int], Awaitable[Any]]) -> Any:
    """Run attempt_fn under the retry/fallback policy. attempt_fn records its own llm_result and raises
    ProviderCallError for provider-side failures (anything else propagates unchanged)."""
    ctx.emitter.emit("llm_route", **decision.public())
    current, tries, n = decision, 0, 0
    while True:
        tries += 1
        n += 1
        try:
            return await attempt_fn(current, n)
        except ProviderCallError as pe:
            if pe.kind in TRANSIENT and tries < MAX_TRIES_PER_PROVIDER and ctx.time_left() > MIN_RETRY_S:
                delay = round(min(4.0, 0.5 * 2 ** (tries - 1)) * random.uniform(0.7, 1.3), 2)
                ctx.emitter.emit("llm_retry", decision_id=current.decision_id, agent=current.agent,
                                 call_site=current.call_site, provider=current.provider, error_kind=pe.kind,
                                 next_attempt=n + 1, backoff_s=delay,
                                 **({"injected": True} if is_injected(pe) else {}))
                await asyncio.sleep(delay)
                continue
            if pe.kind == "budget_exhausted" or (pe.kind == "timeout" and ctx.time_left() < MIN_ATTEMPT_S):
                # The provider is not at fault: the request (or worker) deadline cut the attempt short, so it
                # is not marked failed for the rest of the request and there is no time to fall back.
                raise pe.original
            ctx.failed_providers[current.provider] = pe.kind
            if current is decision and ctx.settings.fallback_enabled and ctx.time_left() > MIN_RETRY_S:
                fb = ctx.router.fallback_decision(decision, pe.kind)
                ctx.fallbacks += 1
                ctx.emitter.emit("fallback", original_decision_id=decision.decision_id,
                                 from_provider=decision.provider, to_provider=fb.provider, error_kind=pe.kind,
                                 attempts_on_original=tries, **fb.public(),
                                 **({"injected": True} if is_injected(pe) else {}))
                current, tries = fb, 0
                continue
            raise pe.original


async def call_llm(ctx, decision: RouteDecision, build: Callable[[Any], Any], inputs: Any) -> Any:
    """Non-streaming routed call. `build(client)` returns the runnable for whichever provider is used
    (so a fallback rebuilds the same chain on the other provider). Returns the runnable's output."""

    async def attempt(dec: RouteDecision, n: int) -> Any:
        t0 = time.perf_counter()
        injected_empty = False

        async def run() -> Any:
            nonlocal injected_empty
            if await maybe_fail(ctx, "provider", dec.provider, agent=dec.agent, call_site=dec.call_site) == "empty":
                injected_empty = True
                return AIMessage(content="", response_metadata={"finish_reason": "stop"})
            return await build(ctx.router.client_for(dec)).ainvoke(inputs)

        try:
            limit = attempt_timeout(ctx, dec)
            out = await asyncio.wait_for(run(), limit)
        except asyncio.CancelledError:
            record_llm_result(ctx, dec, None, t0, status="cancelled", attempt=n)
            raise
        except Exception as e:
            kind = classify_error(e)
            record_llm_result(ctx, dec, None, t0, status="error", attempt=n, error_kind=kind, error=safe_error(e),
                              injected=is_injected(e))
            if is_provider_error(e):
                raise ProviderCallError(kind, e) from e
            raise
        raw = out.get("raw") if isinstance(out, dict) else out
        if is_empty_completion(raw):
            finish = (raw.response_metadata or {}).get("finish_reason")
            e = EmptyResponse(f"{'[injected test fault] ' if injected_empty else ''}empty completion from "
                              f"{dec.provider} (finish_reason={finish})")
            e.injected = injected_empty
            record_llm_result(ctx, dec, raw, t0, status="error", attempt=n, error_kind="empty_response",
                              error=safe_error(e), injected=injected_empty)
            raise ProviderCallError("empty_response", e)
        record_llm_result(ctx, dec, raw if isinstance(raw, AIMessage) else None, t0, attempt=n)
        return out

    return await _resilient(ctx, decision, attempt)


async def stream_llm(ctx, decision: RouteDecision, build: Callable[[Any], Any], inputs: Any,
                     on_delta: Callable[[str], None]) -> str:
    """Streaming routed call: forwards each text delta to on_delta as it arrives and returns the full
    text. Retry/fallback only happen before the first delta; a failure mid-stream is raised."""

    async def attempt(dec: RouteDecision, n: int) -> str:
        t0 = time.perf_counter()
        first_ms, gathered, parts, chunks, max_gap, last = None, None, [], 0, 0, None
        injected_empty = False
        try:
            async with asyncio.timeout(attempt_timeout(ctx, dec)):
                if await maybe_fail(ctx, "provider", dec.provider, agent=dec.agent, call_site=dec.call_site) == "empty":
                    injected_empty = True
                    raise EmptyResponse(f"[injected test fault] empty stream from {dec.provider}")
                async for chunk in build(ctx.router.client_for(dec)).astream(inputs):
                    gathered = chunk if gathered is None else gathered + chunk
                    delta = text_of(chunk.content)
                    if delta:
                        now = time.perf_counter()
                        if first_ms is None:
                            first_ms = int((now - t0) * 1000)
                        else:
                            max_gap = max(max_gap, int((now - last) * 1000))
                        last = now
                        chunks += 1
                        parts.append(delta)
                        on_delta(delta)
        except asyncio.CancelledError:
            record_llm_result(ctx, dec, gathered, t0, status="cancelled", attempt=n, ttft_ms=first_ms)
            raise
        except EmptyResponse as e:
            e.injected = injected_empty
            record_llm_result(ctx, dec, gathered, t0, status="error", attempt=n, error_kind="empty_response",
                              error=safe_error(e), stream_stats={"stream_chunks": 0}, injected=injected_empty)
            raise ProviderCallError("empty_response", e) from e
        except Exception as e:
            kind = classify_error(e)
            record_llm_result(ctx, dec, gathered, t0, status="error", attempt=n, ttft_ms=first_ms,
                              error_kind=kind, error=safe_error(e), stream_stats={"stream_chunks": chunks},
                              injected=is_injected(e))
            if is_provider_error(e) and not parts:
                raise ProviderCallError(kind, e) from e
            raise
        if not parts and not usage_of(gathered)["output_tokens"]:
            # A stream that ends with no text is the streamed form of the empty completion (P5-N1); nothing
            # was forwarded yet, so retry/fallback are still safe.
            e = EmptyResponse(f"empty stream from {dec.provider}")
            record_llm_result(ctx, dec, gathered, t0, status="error", attempt=n, error_kind="empty_response",
                              error=safe_error(e), stream_stats={"stream_chunks": chunks})
            raise ProviderCallError("empty_response", e)
        record_llm_result(ctx, dec, gathered, t0, attempt=n, ttft_ms=first_ms,
                          stream_stats={"stream_chunks": chunks, "max_chunk_gap_ms": max_gap})
        return "".join(parts)

    return await _resilient(ctx, decision, attempt)
