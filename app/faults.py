"""Fault injection for failure-handling tests (Phase 7). Off unless explicitly requested.

A fault spec is a comma-separated list of `kind:name:mode[@n]` rules:

  kind      name                                      modes
  --------  ----------------------------------------  ---------------------------------------------
  worker    research | summary | data | *             timeout, hang, exception, empty
  tool      corpus_search | tavily_search | run_sql | ...  error, timeout, empty
  provider  openai | gemini-openrouter | *            down, timeout, rate_limit, empty, hang

`@n` fails only the first n matching calls (e.g. `worker:research:timeout@1` = fail once, then succeed,
which exercises the retry path); without it every matching call fails. `hang` sleeps until the real
timeout fires, so it exercises the actual timeout and budget code rather than a simulated error.

Where a spec comes from (app.main):
  - per request: the `X-Fault-Inject` header, honoured only when the server has FAULT_INJECT_TOKEN set and
    the request's `X-Fault-Token` header matches it; otherwise the request is refused with 403;
  - process-wide: the FAULT_INJECT env var, for local runs only (scripts/deploy.sh refuses to deploy it).

Every hit is emitted as a `fault_injected` event, and the errors it causes carry `injected: true` in the
trace and are described as "injected test fault" in the answer, so they are never presented as a real
outage. The resulting exceptions carry the same error kinds as real failures, so retry, fallback, skip and
partial-answer handling run unchanged.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

KINDS = {
    "worker": {"timeout", "hang", "exception", "empty"},
    "tool": {"error", "timeout", "empty"},
    "provider": {"down", "timeout", "rate_limit", "empty", "hang"},
}
# The error kind each mode produces (the same kinds real failures use).
ERROR_KIND = {
    ("worker", "timeout"): "timeout",
    ("worker", "exception"): "worker_error",
    ("tool", "error"): "tool_error",
    ("tool", "timeout"): "timeout",
    ("provider", "down"): "provider_unavailable",
    ("provider", "timeout"): "timeout",
    ("provider", "rate_limit"): "rate_limit",
    ("provider", "empty"): "empty_response",
}
MAX_RULES = 8
HANG_S = 3600.0  # longer than any timeout, so the real timeout always fires first


class FaultSpecError(ValueError):
    pass


class InjectedFault(Exception):
    """An error raised on purpose by fault injection. `kind` is the error kind it simulates."""

    injected = True

    def __init__(self, kind: str, target: str):
        super().__init__(f"[injected test fault] {target} -> {kind}")
        self.kind = kind
        self.target = target


def is_injected(e: BaseException | None) -> bool:
    while e is not None:
        if getattr(e, "injected", False):
            return True
        e = getattr(e, "original", None) or e.__cause__
    return False


@dataclass
class Rule:
    kind: str
    name: str
    mode: str
    limit: int | None = None  # None = every call
    hits: int = 0

    @property
    def target(self) -> str:
        return f"{self.kind}:{self.name}:{self.mode}"

    def matches(self, kind: str, name: str) -> bool:
        return self.kind == kind and self.name in (name, "*") and (self.limit is None or self.hits < self.limit)


def parse_spec(spec: str) -> list[Rule]:
    rules = []
    for part in (p.strip() for p in (spec or "").split(",")):
        if not part:
            continue
        body, _, n = part.partition("@")
        bits = body.split(":")
        if len(bits) != 3:
            raise FaultSpecError(f"fault rule {part!r} is not kind:name:mode[@n]")
        kind, name, mode = (b.strip().lower() for b in bits)
        if kind not in KINDS:
            raise FaultSpecError(f"unknown fault kind {kind!r} (use {', '.join(KINDS)})")
        if mode not in KINDS[kind]:
            raise FaultSpecError(f"unknown {kind} fault mode {mode!r} (use {', '.join(sorted(KINDS[kind]))})")
        if not name or len(name) > 40:
            raise FaultSpecError(f"bad fault target name in {part!r}")
        limit = None
        if n:
            if not n.isdigit() or not 1 <= int(n) <= 20:
                raise FaultSpecError(f"fault count in {part!r} must be 1-20")
            limit = int(n)
        rules.append(Rule(kind, name, mode, limit))
    if len(rules) > MAX_RULES:
        raise FaultSpecError(f"at most {MAX_RULES} fault rules")
    return rules


@dataclass
class Faults:
    """The fault rules of one request (never shared between requests) and the hits so far."""

    rules: list[Rule] = field(default_factory=list)
    source: str = ""  # "header" or "env"
    hits: list[dict] = field(default_factory=list)

    @classmethod
    def from_spec(cls, spec: str, source: str) -> "Faults":
        return cls(parse_spec(spec), source)

    @property
    def active(self) -> bool:
        return bool(self.rules)

    def spec(self) -> list[str]:
        return [r.target + (f"@{r.limit}" if r.limit else "") for r in self.rules]

    def take(self, kind: str, name: str) -> Rule | None:
        for r in self.rules:
            if r.matches(kind, name):
                r.hits += 1
                return r
        return None


async def maybe_fail(ctx, kind: str, name: str, **where) -> str | None:
    """Apply the first matching rule for this call. Raises InjectedFault for error modes, sleeps for
    `hang` (the caller's real timeout fires), and returns "empty" when the caller should produce an empty
    result instead. Returns None when no rule matches (the normal case)."""
    faults: Faults | None = getattr(ctx, "faults", None)
    if faults is None or not faults.active:
        return None
    rule = faults.take(kind, name)
    if rule is None:
        return None
    hit = {"target": rule.target, "kind": kind, "name": name, "mode": rule.mode, "hit": rule.hits,
           "source": faults.source, **where}
    faults.hits.append(hit)
    ctx.emitter.emit("fault_injected", injected=True, **hit,
                     note="Deliberate test fault (fault injection), not a real outage.")
    if rule.mode == "empty":
        return "empty"
    if rule.mode == "hang":
        await asyncio.sleep(HANG_S)
        return None
    raise InjectedFault(ERROR_KIND[(kind, rule.mode)], f"{kind}:{name}")
