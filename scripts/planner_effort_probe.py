#!/usr/bin/env python3
"""Planner reasoning-effort probe (Phase 9 decision input): `low` vs `minimal` for the Gemini planner.

Runs the real planner node (Gemini via OpenRouter) on the tester's 12 held-out Phase 6 phrasings plus the
7 Phase 8b latency questions, `--runs` times per effort, in one event loop. Records per call: plan worker
classes (checked against the expected class), dependency shape, the papers/queries the planner named, the
seeds that resolved, planner latency, output and reasoning tokens, and cost. No retriever is attached, so
no speculative search or embedding call is made. Writes OUT_DIR/planner_effort.jsonl and summary.json.

Usage: .venv/bin/python scripts/planner_effort_probe.py OUT_DIR [--runs 2] [--efforts low minimal]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASES = [  # (question, expected class) - held-out phrasings from tests/integration/test_phase6_tester_planner_live.py
    ("What fraction of the indexed papers have an accompanying code release?", "data"),
    ("List every paper in the collection whose primary arXiv category is cs.IR.", "data"),
    ("Which author appears on the most papers here?", "data"),
    ("When was the oldest paper in the collection first posted to arXiv?", "data"),
    ("Break down the corpus by topic group, with counts.", "data"),
    ("Did anyone named Wang co-author more than one indexed paper?", "data"),
    ("What is the main limitation of tree-of-thoughts search as discussed in the papers?", "content"),
    ("Explain the retrieval critique tokens used in Self-RAG.", "content"),
    ("How do the papers evaluate multi-agent debate against single-agent baselines?", "content"),
    ("Grab the three newest papers tagged cs.MA and tell me what each one contributes.", "mixed"),
    ("Among the papers that ship code, which ones from 2023 tackle tool use, and how do they work?", "mixed"),
    ("Give a quick summary of whatever the earliest 2024 paper in the index is.", "mixed"),
    # Phase 8b latency questions (scripts/measure_latency.py)
    ("How does Self-RAG decide when to retrieve?", "content"),
    ("Summarize the ReAct paper.", "content"),
    ("How do RAG systems handle noisy or irrelevant retrieved passages?", "content"),
    ("How do LLM agents use memory to handle long conversations?", "content"),
    ("What environments does AgentBench use to evaluate LLM agents?", "content"),
    ("Which 2025 cs.MA papers are in the corpus? Summarize the two most recent.", "mixed"),
    ("Compare Self-RAG and Corrective RAG (CRAG): how does each handle bad retrievals?", "content"),
]


def correct(want: str, tasks: list[dict]) -> bool:
    workers = [t["worker"] for t in tasks]
    if want == "data":
        return workers[:1] == ["data"]
    if want == "content":
        return "data" not in workers and bool(workers)
    first = tasks[0]
    rest = tasks[1:]
    return first["worker"] == "data" and any(t["worker"] in ("research", "summary") and first["id"] in t["depends_on"]
                                             for t in rest)


async def one(effort: str, q: str, want: str) -> dict:
    os.environ["GEMINI_PLANNER_REASONING_EFFORT"] = effort
    from app.config import load_settings
    from app.context import RequestContext
    from app.events import Emitter
    from app.graph import planner
    from app.llm.router import Router

    s = load_settings()
    em = Emitter("probe")
    ctx = RequestContext(request_id="probe", query=q, k=s.top_k, settings=s, router=Router(s),
                         retriever=None, emitter=em, tavily=None)
    out = await planner({"query": q}, {"configurable": {"ctx": ctx}})
    events = []
    while not em.queue.empty():
        events.append(em.queue.get_nowait())
    res = [p for e, p in events if e == "llm_result" and p.get("call_site") == "planner"]
    plan = next((p for e, p in events if e == "plan"), {})
    ok_calls = [r for r in res if r["status"] == "ok"]
    last = ok_calls[-1] if ok_calls else (res[-1] if res else {})
    tasks = out["tasks"]
    return {"effort": effort, "q": q, "want": want, "plan_source": out["plan_source"],
            "correct": out["plan_source"] == "llm" and correct(want, tasks),
            "tasks": [{"id": t["id"], "worker": t["worker"], "depends_on": t["depends_on"],
                       "needs_live": t["needs_live"], "papers": t.get("papers", []), "queries": t.get("queries", []),
                       "instruction": t["instruction"]} for t in tasks],
            "seeds": [sd for t in plan.get("tasks", []) for sd in t.get("seeds", [])],
            "notes": plan.get("notes", []), "attempts": len(res),
            "latency_ms": last.get("latency_ms"), "output_tokens": last.get("output_tokens"),
            "reasoning_tokens": last.get("reasoning_tokens"), "cost_usd": sum(r.get("cost_usd") or 0 for r in res)}


async def main(out_dir: Path, runs: int, efforts: list[str]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with (out_dir / "planner_effort.jsonl").open("w") as f:
        for r in range(runs):
            for q, want in CASES:
                for eff in efforts:  # interleaved so drift in provider latency hits both efforts alike
                    row = {"run": r + 1, **(await one(eff, q, want))}
                    rows.append(row)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(f"{eff:8s} {row['correct']!s:5s} {row['latency_ms']}ms out={row['output_tokens']} "
                          f"reason={row['reasoning_tokens']} {[t['worker'] for t in row['tasks']]} {q[:50]}", flush=True)
    summary = {}
    for eff in efforts:
        rs = [x for x in rows if x["effort"] == eff]
        lat = [x["latency_ms"] for x in rs if x["latency_ms"] is not None]
        summary[eff] = {"n": len(rs), "correct": sum(x["correct"] for x in rs), "fallback": sum(x["plan_source"] != "llm" for x in rs),
                        "latency_ms_median": statistics.median(lat), "latency_ms_p90": sorted(lat)[int(0.9 * (len(lat) - 1))],
                        "latency_ms_max": max(lat),
                        "reasoning_calls": sum(1 for x in rs if (x["reasoning_tokens"] or 0) > 0),
                        "output_tokens_median": statistics.median([x["output_tokens"] or 0 for x in rs]),
                        "summary_with_paper_seed": sum(1 for x in rs if any(s["tool"] != "corpus_search" for s in x["seeds"])),
                        "cost_usd": round(sum(x["cost_usd"] for x in rs), 4)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--efforts", nargs="+", default=["low", "minimal"])
    a = ap.parse_args()
    sys.exit(asyncio.run(main(Path(a.out_dir), a.runs, a.efforts)))
