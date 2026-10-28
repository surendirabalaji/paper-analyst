"""Per-request context passed to graph nodes through the LangGraph config (never shared across requests)."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.events import Emitter
from app.faults import Faults
from app.llm.router import RouteDecision, Router
from app.mcp_tavily import TavilyMCP
from app.rag import EvidenceRegistry, Retriever


SPEC_DISCARD_WAIT_S = 1.0


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
    # Provider errors no injected fault caused (P8b-D2): "provider: kind". A task that failed while a fault
    # was injected is labelled injected only if none of these happened during it.
    real_provider_errors: list[str] = field(default_factory=list)
    # Speculative retrieval (Phase 8b): {"query", "k", "task", "status"}; the task returns (chunks, ms).
    speculative: dict[str, Any] | None = None

    def start_speculative(self, query: str, k: int) -> None:
        """Start the corpus search of the question while the planner plans (no citation ids yet)."""

        async def run():
            t0 = time.perf_counter()
            chunks = await self.retriever.search(query, k)
            return chunks, int((time.perf_counter() - t0) * 1000)

        task = asyncio.create_task(run())
        task.add_done_callback(lambda t: t.cancelled() or t.exception())  # never an unretrieved exception
        self.speculative = {"query": query, "k": k, "task": task, "status": "running"}

    async def end_speculative(self, used: bool, reason: str) -> None:
        """Keep the speculative result for a corpus_search seed, or discard it; traced either way. A discarded
        search is not cancelled mid-request (that would break its pooled HTTP connection): it started with the
        planner, so it has normally finished already; it is cancelled only if it is still running after 1 s."""
        spec = self.speculative
        if not spec or spec["status"] != "running":
            return
        spec["status"] = "kept" if used else "discarded"
        if not used:
            try:
                await asyncio.wait_for(asyncio.shield(spec["task"]), SPEC_DISCARD_WAIT_S)
            except asyncio.TimeoutError:
                spec["task"].cancel()
            except Exception:  # a failed discarded search does not matter
                pass
        self.emitter.emit("speculative_retrieval", status=spec["status"], query=spec["query"], k=spec["k"],
                          reason=reason)

    async def take_speculative(self, query: str, k: int):
        """(chunks, ms) if a kept speculative search matches this corpus_search call, else None (used once)."""
        spec = self.speculative
        if not spec or spec["status"] != "kept" or spec["query"] != query or spec["k"] != k:
            return None
        spec["status"] = "used"
        try:
            return await spec["task"]
        except Exception:  # the search failed: the caller searches normally and reports any error itself
            return None

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
