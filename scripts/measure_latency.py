#!/usr/bin/env python3
"""Latency measurement for Phase 8b: stream a fixed question set against a deployed (or local) URL.

For every run it records client-side wall-clock times (first SSE byte, TTFT = first `token` event, total),
the server's own `done.timings`, the plan (workers), the providers used and the estimated cost, and saves
the raw SSE events as JSONL. A summary JSON (per-category medians) is written next to the raw files.

Usage:
  .venv/bin/python scripts/measure_latency.py URL OUT_DIR [--runs N] [--category simple|multi|all]
                                              [--warmup] [--only "question substring"]

Categories: `simple` = questions a single worker should answer; `multi` = questions whose plan is
expected to use two or more workers. The category is fixed per question in QUESTIONS, not inferred from
the plan, so the medians compare like with like before and after a change. Nothing secret is sent or saved.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

QUESTIONS = [
    ("simple", "self_rag_retrieve", "How does Self-RAG decide when to retrieve?"),
    ("simple", "react_summary", "Summarize the ReAct paper."),
    ("simple", "noisy_passages", "How do RAG systems handle noisy or irrelevant retrieved passages?"),
    ("simple", "agent_memory", "How do LLM agents use memory to handle long conversations?"),
    ("simple", "agentbench_envs", "What environments does AgentBench use to evaluate LLM agents?"),
    ("multi", "cs_ma_2025", "Which 2025 cs.MA papers are in the corpus? Summarize the two most recent."),
    ("multi", "selfrag_vs_crag", "Compare Self-RAG and Corrective RAG (CRAG): how does each handle bad retrievals?"),
]


def run_one(client: httpx.Client, url: str, query: str, raw_path: Path) -> dict:
    events: list[dict] = []
    first_byte = first_token = None
    event = None
    t0 = time.perf_counter()
    with client.stream("POST", f"{url}/api/chat", json={"query": query}) as r:
        status = r.status_code
        for line in r.iter_lines():
            now = time.perf_counter() - t0
            if first_byte is None:
                first_byte = now
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                try:
                    data = json.loads(line[5:])
                except ValueError:
                    data = {"_unparsed": line[5:][:200]}
                events.append({"event": event, "client_s": round(now, 3), "data": data})
                if event == "token" and first_token is None:
                    first_token = now
    total = time.perf_counter() - t0
    raw_path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n", encoding="utf-8")
    done = next((e["data"] for e in reversed(events) if e["event"] == "done"), {})
    plan = next((e["data"] for e in events if e["event"] == "plan"), {})
    routes = [e["data"] for e in events if e["event"] == "llm_route"]
    return {
        "http_status": status,
        "first_byte_s": round(first_byte or -1, 3),
        "ttft_s": round(first_token or -1, 3),
        "total_s": round(total, 3),
        "server_timings": done.get("timings"),
        "status": done.get("status"),
        "workers": [t.get("worker") for t in plan.get("tasks", [])],
        "providers_used": done.get("providers_used"),
        "llm_calls": [f"{r.get('agent')}:{r.get('call_site')}:{r.get('provider')}" for r in routes],
        "cost_usd": done.get("cost_usd"),
        "n_citations": sum(1 for e in events if e["event"] == "citation"),
        "n_tokens": sum(1 for e in events if e["event"] == "token"),
        "errors": [e["data"].get("message_safe", "")[:160] for e in events if e["event"] == "error"],
        "raw": raw_path.name,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out_dir")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--category", default="all", choices=["simple", "multi", "all"])
    ap.add_argument("--warmup", action="store_true", help="hit /health and run one unrecorded question first")
    ap.add_argument("--only", default="", help="run only questions whose text contains this substring")
    a = ap.parse_args()
    url = a.url.rstrip("/")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    qs = [q for q in QUESTIONS if (a.category == "all" or q[0] == a.category) and a.only.lower() in q[2].lower()]
    results = []
    with httpx.Client(timeout=httpx.Timeout(300.0, connect=60.0)) as client:
        t0 = time.perf_counter()
        h = client.get(f"{url}/health")
        health = {"status": h.status_code, "latency_s": round(time.perf_counter() - t0, 3)}
        if a.warmup:
            run_one(client, url, QUESTIONS[0][2], out / "warmup.events.jsonl")
        for i in range(a.runs):
            for cat, key, q in qs:
                r = run_one(client, url, q, out / f"{key}.run{i + 1}.events.jsonl")
                r.update({"category": cat, "key": key, "query": q, "run": i + 1,
                          "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                results.append(r)
                print(json.dumps({k: r[k] for k in ("category", "key", "run", "ttft_s", "total_s", "status",
                                                    "workers", "providers_used", "cost_usd", "errors")}),
                      flush=True)
    summary: dict = {"url": url, "health": health, "warmup": a.warmup, "runs": a.runs, "categories": {}}
    for cat in ("simple", "multi"):
        rs = [r for r in results if r["category"] == cat and r["ttft_s"] > 0]
        if rs:
            summary["categories"][cat] = {
                "n": len(rs),
                "ttft_median_s": round(statistics.median(r["ttft_s"] for r in rs), 3),
                "ttft_min_s": min(r["ttft_s"] for r in rs),
                "ttft_max_s": max(r["ttft_s"] for r in rs),
                "total_median_s": round(statistics.median(r["total_s"] for r in rs), 3),
                "cost_sum_usd": round(sum(r["cost_usd"] or 0 for r in rs), 4),
            }
    (out / "results.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
