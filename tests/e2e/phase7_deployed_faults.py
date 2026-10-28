"""Phase 7 deployed fault-injection driver (REAL deployment, real providers; costs about $0.02-0.06 per case).

Usage: .venv/bin/python tests/e2e/phase7_deployed_faults.py <base_url> [case ...]
Reads FAULT_INJECT_TOKEN from .env (never printed). Saves each case's SSE events to
eval/results/phase7/deployed/<case>.events.jsonl and prints a short summary.

Cases:
  partial   two-part question, content worker (research/summary) fails (exception)  -> partial answer
  retry     Data-only question, Data times out once (timeout@1)                        -> retry, then done
  empty     Gemini returns empty completions twice (provider:gemini-openrouter:empty@2) -> retry, fallback
  guard     the same fault header without the token                                   -> 403
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import httpx
from dotenv import dotenv_values

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "eval" / "results" / "phase7" / "deployed"
CASES = {
    "partial": ("How many papers in the corpus have cs.MA as a category, and how does Self-RAG decide when to retrieve?",
                "worker:research:exception,worker:summary:exception"),
    "retry": ("Which authors appear on more than one paper in the corpus?", "worker:data:timeout@1"),
    "empty": ("How does Self-RAG decide when to retrieve?", "provider:gemini-openrouter:empty@2"),
}


def parse(text: str):
    out = []
    for frame in text.split("\n\n"):
        ev, data = None, None
        for line in frame.splitlines():
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if ev:
            out.append((ev, data))
    return out


def run(base: str, name: str, token: str) -> dict:
    if name == "guard":
        r = httpx.post(f"{base}/api/chat", json={"query": CASES["partial"][0]},
                       headers={"X-Fault-Inject": CASES["partial"][1]}, timeout=30)
        return {"case": name, "http": r.status_code, "body": r.text[:120]}
    query, spec = CASES[name]
    t0 = time.perf_counter()
    r = httpx.post(f"{base}/api/chat", json={"query": query}, timeout=300,
                   headers={"X-Fault-Inject": spec, "X-Fault-Token": token})
    wall = round(time.perf_counter() - t0, 1)
    assert token not in r.text, "token echoed"
    events = parse(r.text)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.events.jsonl").write_text(
        "".join(json.dumps({"event": e, "data": d}, default=str) + "\n" for e, d in events))
    done = next((d for e, d in events if e == "done"), {})
    answer = "".join(d["delta"] for e, d in events if e == "token")
    return {
        "case": name, "http": r.status_code, "wall_s": wall, "request_id": done.get("request_id"),
        "status": done.get("status"), "last_event": events[-1][0] if events else None,
        "plan": [t["worker"] for d in [next((d for e, d in events if e == "plan"), {"tasks": []})] for t in d["tasks"]],
        "tasks": done.get("tasks"),
        "faults": [h["target"] for h in done.get("faults_injected", [])],
        "retries": [d["reason"] for e, d in events if e == "transition" and d["reason"].startswith(("retry", "not retrying"))],
        "llm_retry": len([1 for e, _ in events if e == "llm_retry"]),
        "fallbacks": [(d["call_site"], d["from_provider"], d["to_provider"], d.get("injected")) for e, d in events
                      if e == "fallback"],
        "citations": [d["cite_id"] for e, d in events if e == "citation"],
        "nonrecoverable_errors": [d for e, d in events if e == "error" and not d.get("recoverable")],
        "over_budget_s": done.get("over_budget_s"), "cost_usd": done.get("cost_usd"),
        "ttft_ms": (done.get("timings") or {}).get("ttft_ms"), "total_ms": (done.get("timings") or {}).get("total_ms"),
        "answer": answer,
    }


def main() -> None:
    base = sys.argv[1].rstrip("/")
    names = sys.argv[2:] or ["guard", "partial", "retry", "empty"]
    token = dotenv_values(ROOT / ".env").get("FAULT_INJECT_TOKEN") or ""
    if not token:
        sys.exit("FAULT_INJECT_TOKEN missing in .env")
    results = [run(base, n, token) for n in names]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(results, indent=1, default=str))
    for res in results:
        print(json.dumps(res, indent=1, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
