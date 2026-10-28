"""Per-request context passed to graph nodes through the LangGraph config (never shared across requests)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.config import Settings
from app.events import Emitter
from app.faults import Faults
from app.llm.router import RouteDecision, Router
from app.mcp_tavily import TavilyMCP
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
    tavily: TavilyMCP | None = None  # process-wide MCP handle; each call opens its own session
    started: float = field(default_factory=time.perf_counter)
    retrieval_ms_sum: int = 0
    llm_ms_sum: int = 0
    ttft_ms: int | None = None
    tokens_by_provider: dict = field(default_factory=dict)
    failed_providers: dict = field(default_factory=dict)  # provider -> error kind, this request only
    fallbacks: int = 0
    cost_basis: dict = field(default_factory=dict)  # "reported" / "estimated" -> number of priced calls
    faults: Faults = field(default_factory=Faults)  # fault injection rules for this request (normally none)
    stage_deadline: float | None = None  # perf_counter deadline of the running worker, if any

    def route(self, call_site: str, agent: str, features: dict | None = None) -> RouteDecision:
        """Routing decision for one LLM call; providers that already failed in this request are avoided."""
        return self.router.decide(call_site, agent, features, failed=self.failed_providers)

    def remaining_s(self) -> float:
        """Seconds left in the whole request budget (REQUEST_BUDGET_S)."""
        return self.settings.request_budget_s - (time.perf_counter() - self.started)

    def synth_reserve_s(self) -> float:
        """Request time kept back so synthesis can still run: SYNTH_RESERVE_S, at most a quarter of the budget."""
        return min(self.settings.synth_reserve_s, 0.25 * self.settings.request_budget_s)

    def worker_window_s(self) -> float:
        """Seconds a worker may still use: the request budget left minus the synthesis reserve."""
        return self.remaining_s() - self.synth_reserve_s()

    def providers_down(self) -> bool:
        """Both generation providers already failed in this request (a worker retry would be wasted)."""
        return len(self.failed_providers) >= 2

    def time_left(self) -> float:
        """Seconds left for the current stage: the request budget, or the running worker's deadline if that
        is sooner. Every LLM attempt timeout is capped by this (P2-N7 / P4-N1)."""
        left = self.remaining_s()
        if self.stage_deadline is not None:
            left = min(left, self.stage_deadline - time.perf_counter())
        return left


def ctx_of(config) -> RequestContext:
    return config["configurable"]["ctx"]
