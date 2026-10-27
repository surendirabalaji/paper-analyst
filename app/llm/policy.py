"""Hybrid cost/capability routing policy (pure functions, no I/O).

Every generation call is routed from features computed in code (never by an LLM):

  call site          | rule                                                        | dimension  | provider
  -------------------+-------------------------------------------------------------+------------+------------------
  planner            | always                                                      | cost       | Gemini/OpenRouter
  worker_tool_loop / | >= 3k tokens of full paper sections read, or >= 10k tokens  | capability | OpenAI
  worker_final       | of context (faithful condensation of long source text)      |            |
  worker_tool_loop   | otherwise (tool choice and query writing, short outputs)    | cost       | Gemini/OpenRouter
  worker_final       | otherwise (findings from evidence already in context)       | cost       | Gemini/OpenRouter
  synthesize_no_task | greeting / off-scope reply, no evidence                     | cost       | Gemini/OpenRouter
  synthesize         | >= 2 completed tasks, or >= 3 distinct papers, or a          | capability | OpenAI
                     | comparison question over >= 2 papers, or evidence >= 6k tok  |            |
  synthesize         | otherwise (one task, one or two papers, short evidence)     | cost       | Gemini/OpenRouter
  any                | the chosen provider already failed in this request          | availability | the other one
  any                | ROUTER_FORCE_PROVIDER set (evaluation runs only)            | forced     | that provider

The cost side is Gemini 3.8 Flash ($1.50 / $7.50 per 1M in/out as billed through OpenRouter BYOK) and the
capability side is GPT-6.1 Sol ($2 / $10), from PRICES below. Reactive fallback after an error is handled in router.py and traced
as a separate `fallback` event; it is not part of this table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

OPENAI = "openai"
GEMINI = "gemini-openrouter"
PROVIDERS = (OPENAI, GEMINI)

# Per-1M-token prices, verified 2026-10-27. Keyed by exact model id so a model change without a price
# update gives est_cost_usd=None instead of a wrong number.
PRICES = {
    "gpt-6.1-sol": {
        "input": 2.00, "cached_input": 0.10, "output": 10.00,
        "source": "https://developers.openai.com/api/docs/pricing (standard tier, short context)",
        "checked": "2026-10-27",
    },
    # Gemini calls go through OpenRouter with the student's own Google key (BYOK), so the OpenRouter catalog
    # price ($0.825 / $4.125) is not what is charged. OpenRouter's reported upstream cost matched $1.50 / $7.50
    # exactly on every measured call (tester P4-D1). Google's page lists that as the standard rate from
    # 2027-01-01 and a lower $0.75 / $3.75 through 2026-12-31; until the Google bill confirms which applies,
    # we use the conservative (higher) reported rate.
    "google/gemini-3.8-flash": {
        "input": 1.50, "cached_input": 0.15, "output": 7.50,
        "source": "https://ai.google.dev/gemini-api/docs/pricing (standard tier from 2027-01-01; matches "
                  "OpenRouter-reported BYOK upstream cost)",
        "checked": "2026-10-27",
    },
}

# Thresholds for the capability side of synthesis.
SYNTH_MIN_TASKS = 2
SYNTH_MIN_PAPERS = 3
SYNTH_COMPARE_MIN_PAPERS = 2
SYNTH_MIN_EVIDENCE_TOKENS = 6000

# Thresholds for the capability side of worker calls (P4-N2). Full-section text comes from Summary's
# get_paper_section; a short outline or a handful of search passages stays well below both.
WORKER_LONG_SOURCE_TOKENS = 3000
WORKER_LONG_CONTEXT_TOKENS = 10000

# Per-attempt timeouts by call site (seconds). Structured calls are short; synthesis streams longer.
TIMEOUTS = {"planner": 20.0, "worker_tool_loop": 25.0, "worker_final": 25.0}

_COMPARE = re.compile(
    r"\b(compare[sd]?|comparison|comparing|versus|vs\.?|differ(s|ence|ences)?|contrast|trade-?offs?|"
    r"similarit(y|ies)|relative to|better than|worse than|which (is|one) (better|more)|"
    r"stacks? up|outperform(s|ed)?|head[- ]to[- ]head|as opposed to|unlike|distinguish(es)? between)\b",
    re.I,
)


def comparison_cue(query: str) -> bool:
    return bool(_COMPARE.search(query or ""))


def approx_tokens(text: str) -> int:
    """Cheap token estimate (about 4 characters per token for English); labelled *_est in the trace."""
    return (len(text or "") + 3) // 4


def other(provider: str) -> str:
    return GEMINI if provider == OPENAI else OPENAI


@dataclass(frozen=True)
class Choice:
    provider: str
    dimension: str  # cost | capability | availability | forced
    rationale: str


def _synth_reasons(f: dict) -> list[str]:
    reasons = []
    if f.get("n_tasks_ok", 0) >= SYNTH_MIN_TASKS:
        reasons.append(f"{f['n_tasks_ok']} worker results to reconcile")
    if f.get("n_papers", 0) >= SYNTH_MIN_PAPERS:
        reasons.append(f"evidence spans {f['n_papers']} papers")
    if f.get("comparison") and f.get("n_papers", 0) >= SYNTH_COMPARE_MIN_PAPERS:
        reasons.append("comparison question across papers")
    if f.get("evidence_tokens_est", 0) >= SYNTH_MIN_EVIDENCE_TOKENS:
        reasons.append(f"long evidence (~{f['evidence_tokens_est']} tokens)")
    return reasons


def _worker_reasons(f: dict) -> list[str]:
    reasons = []
    if f.get("long_source_tokens_est", 0) >= WORKER_LONG_SOURCE_TOKENS:
        reasons.append(f"~{f['long_source_tokens_est']} tokens of full paper sections in context")
    if f.get("context_tokens_est", 0) >= WORKER_LONG_CONTEXT_TOKENS:
        reasons.append(f"~{f['context_tokens_est']} tokens of context")
    return reasons


def base_choice(call_site: str, f: dict) -> Choice:
    """The policy table, before availability and force overrides."""
    if call_site == "planner":
        return Choice(GEMINI, "cost", "Short structured planning (JSON task list); the low-cost model is sufficient.")
    if call_site in ("worker_tool_loop", "worker_final"):
        reasons = _worker_reasons(f)
        if reasons:
            return Choice(OPENAI, "capability", "Faithful condensation of long source text needs the stronger "
                                                "model: " + "; ".join(reasons) + ".")
    if call_site == "worker_tool_loop":
        return Choice(GEMINI, "cost", "Tool selection and search-query writing with short outputs; "
                                      "the low-cost, fast model is sufficient.")
    if call_site == "worker_final":
        return Choice(GEMINI, "cost", "Compact findings from evidence already retrieved; low-cost model.")
    if call_site == "synthesize_no_task":
        return Choice(GEMINI, "cost", "Short reply with no evidence to reconcile; low-cost model.")
    if call_site == "synthesize":
        reasons = _synth_reasons(f)
        if reasons:
            return Choice(OPENAI, "capability", "Multi-source synthesis needs the stronger model: " + "; ".join(reasons) + ".")
        n = f.get("n_papers", 0)
        return Choice(GEMINI, "cost", f"Grounded answer from {n} paper{'s' if n != 1 else ''} and one worker result; "
                                      "the low-cost model is sufficient.")
    return Choice(GEMINI, "cost", f"Unclassified call site {call_site!r}; default to the low-cost model.")


def choose(call_site: str, features: dict, *, failed: dict[str, str] | None = None, force: str = "") -> Choice:
    """Final routing choice. `failed` maps provider -> error kind for providers that already failed in
    this request; `force` is the evaluation-only ROUTER_FORCE_PROVIDER override."""
    if force in PROVIDERS:
        return Choice(force, "forced", f"ROUTER_FORCE_PROVIDER={force} (evaluation override); policy bypassed.")
    c = base_choice(call_site, features)
    failed = failed or {}
    if c.provider in failed and other(c.provider) not in failed:
        return Choice(other(c.provider), "availability",
                      f"Policy picked {c.provider} ({c.dimension}), but it already failed in this request "
                      f"({failed[c.provider]}); using {other(c.provider)}.")
    return c


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> float | None:
    p = PRICES.get(model_id)
    if not p:
        return None
    fresh = max(0, input_tokens - cached_input_tokens)
    usd = (fresh * p["input"] + cached_input_tokens * p["cached_input"] + output_tokens * p["output"]) / 1_000_000
    return round(usd, 6)
