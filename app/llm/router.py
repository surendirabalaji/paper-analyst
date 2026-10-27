"""Generation-provider router.

Every LLM call goes through `Router.decide()` and is traced as an `llm_route` event (decision) plus
an `llm_result` event (tokens, latency). Phase 2 has no routing policy yet: every call uses
DEFAULT_PROVIDER and the rationale says so. Phase 4 replaces `decide()` with the hybrid
cost/capability policy; the call sites and trace format stay the same.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI

from app.config import Settings

PROVIDER_LABELS = {
    "openai": "OpenAI (direct API)",
    "gemini-openrouter": "Google Gemini (served through OpenRouter)",
}


@dataclass
class RouteDecision:
    call_site: str
    agent: str
    provider: str
    model_id: str
    dimension: str
    rationale: str
    features: dict[str, Any] = field(default_factory=dict)
    decision_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d["provider_label"] = PROVIDER_LABELS.get(self.provider, self.provider)
        return d


def text_of(content: Any) -> str:
    """Message content as plain text (content may be a str or a list of content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "")


def usage_of(msg: BaseMessage | None) -> dict[str, int]:
    u = getattr(msg, "usage_metadata", None) or {}
    return {"input_tokens": int(u.get("input_tokens", 0)), "output_tokens": int(u.get("output_tokens", 0))}


class Router:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._clients: dict[str, ChatOpenAI] = {}

    def model_id(self, provider: str) -> str:
        return self.settings.openai_model if provider == "openai" else self.settings.gemini_model

    def client(self, provider: str) -> ChatOpenAI:
        if provider not in self._clients:
            s = self.settings
            common = dict(stream_usage=True, timeout=s.llm_timeout_s, max_retries=1)
            if provider == "openai":
                self._clients[provider] = ChatOpenAI(model=s.openai_model, api_key=s.openai_api_key, **common)
            elif provider == "gemini-openrouter":
                self._clients[provider] = ChatOpenAI(
                    model=s.gemini_model,
                    base_url=s.openrouter_base_url,
                    api_key=s.openrouter_api_key,
                    extra_body={"usage": {"include": True}},
                    **common,
                )
            else:
                raise ValueError(f"unknown provider {provider!r}")
        return self._clients[provider]

    def decide(self, call_site: str, agent: str, features: dict[str, Any] | None = None) -> RouteDecision:
        provider = self.settings.default_provider
        return RouteDecision(
            call_site=call_site,
            agent=agent,
            provider=provider,
            model_id=self.model_id(provider),
            dimension="fixed default",
            rationale="Phase 2 build: every call uses the configured default provider; "
            "cost/capability routing is not implemented yet.",
            features=features or {},
        )


async def traced_invoke(ctx, decision: RouteDecision, runnable, inputs) -> Any:
    """Invoke a runnable built on the decision's client and trace it. Returns the runnable's output."""
    ctx.emitter.emit("llm_route", **decision.public())
    t0 = time.perf_counter()
    out = await runnable.ainvoke(inputs)
    raw = out.get("raw") if isinstance(out, dict) else out
    record_llm_result(ctx, decision, raw if isinstance(raw, AIMessage) else None, t0)
    return out


def record_llm_result(ctx, decision: RouteDecision, msg: BaseMessage | None, t0: float, ttft_ms: int | None = None):
    latency_ms = int((time.perf_counter() - t0) * 1000)
    usage = usage_of(msg)
    ctx.llm_ms_sum += latency_ms
    tb = ctx.tokens_by_provider.setdefault(decision.provider, {"input_tokens": 0, "output_tokens": 0, "calls": 0})
    tb["input_tokens"] += usage["input_tokens"]
    tb["output_tokens"] += usage["output_tokens"]
    tb["calls"] += 1
    model_returned = (getattr(msg, "response_metadata", None) or {}).get("model_name")
    ctx.emitter.emit(
        "llm_result",
        decision_id=decision.decision_id,
        agent=decision.agent,
        call_site=decision.call_site,
        provider=decision.provider,
        model_returned=model_returned,
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        **usage,
    )
