"""Per-request context passed to graph nodes through the LangGraph config (never shared across requests)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.config import Settings
from app.events import Emitter
from app.llm.router import RouteDecision, Router
from app.rag import EvidenceRegistry, Retriever


@dataclass
class RequestContext:
    request_id: str
    query: str
    k: int
    settings: Settings
    router: Router
    retriever: Retriever
    emitter: Emitter
    registry: EvidenceRegistry = field(default_factory=EvidenceRegistry)
    started: float = field(default_factory=time.perf_counter)
    retrieval_ms_sum: int = 0
    llm_ms_sum: int = 0
    ttft_ms: int | None = None
    tokens_by_provider: dict = field(default_factory=dict)
    failed_providers: dict = field(default_factory=dict)  # provider -> error kind, this request only
    fallbacks: int = 0
    cost_basis: dict = field(default_factory=dict)  # "reported" / "estimated" -> number of priced calls

    def route(self, call_site: str, agent: str, features: dict | None = None) -> RouteDecision:
        """Routing decision for one LLM call; providers that already failed in this request are avoided."""
        return self.router.decide(call_site, agent, features, failed=self.failed_providers)

    def remaining_s(self) -> float:
        return self.settings.request_budget_s - (time.perf_counter() - self.started)


def ctx_of(config) -> RequestContext:
    return config["configurable"]["ctx"]
