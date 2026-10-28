#!/usr/bin/env python3
"""Aggregate Phase 9 results into eval/results/phase9/summary.json (no network, no LLM calls).

Inputs (written by run_eval.py, measure_latency.py, ragas_eval.py, check_system.py):
  routed-deployed/            main configuration: deployed rev 00018, normal routing, warm, all 44 items once
  routed-deployed-multi/      the two Data -> Summary items, 3 more runs each (warm)
  forced-openai-local/        local server, ROUTER_FORCE_PROVIDER=openai, 32 RAGAS items
  forced-gemini-local/        local server, ROUTER_FORCE_PROVIDER=gemini-openrouter, 32 RAGAS items
  cold-start-1/, cold-start-2/

Latency fields per run: client TTFT (first `token` event) and total from the client clock; retrieval_ms and
llm_ms_sum from the server's `done.timings`. llm_ms_sum adds up every LLM call; calls in one request run one
after another except the speculative corpus search, which overlaps the planner, so retrieval + LLM can
exceed total by that overlap and the parts must not be added up as if sequential.
Cost: the per-request `cost_usd` the server computes from recorded tokens (OpenRouter-reported cost for
Gemini, dated price table for OpenAI; app/llm/policy.py, checked 2026-10-27): an estimate, not a bill.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P9 = ROOT / "eval/results/phase9"
TESTS = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "eval/testset/testset_v1.jsonl").read_text().splitlines() if l.strip()}


def pct(vals, q):
    s = sorted(vals)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


def stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"n": len(vals), "median": round(statistics.median(vals), 3), "p90": round(pct(vals, 0.9), 3),
            "min": round(min(vals), 3), "max": round(max(vals), 3), "mean": round(statistics.mean(vals), 3)}


def results(d):
    p = P9 / d / "results.json"
    return json.loads(p.read_text())["results"] if p.exists() else []


def lat_block(rows):
    st = [r.get("server_timings") or {} for r in rows]
    return {"ttft_s": stats([r.get("ttft_s") for r in rows]),
            "total_s": stats([r.get("total_s") for r in rows]),
            "retrieval_s": stats([t.get("retrieval_ms", 0) / 1000 for t in st if t]),
            "llm_sum_s": stats([t.get("llm_ms_sum", 0) / 1000 for t in st if t]),
            "server_ttft_s": stats([t.get("ttft_ms", 0) / 1000 for t in st if t]),
            "cost_usd": stats([r.get("cost_usd") for r in rows]),
            "status": {s: sum(1 for r in rows if r.get("status") == s) for s in sorted({r.get("status") for r in rows})}}


def provider_share(rows):
    calls = [c for r in rows for c in r.get("llm_calls", []) if isinstance(c, dict) and c.get("status") == "ok"]
    by = {}
    for c in calls:
        b = by.setdefault(c["provider"], {"calls": 0, "by_call_site": {}})
        b["calls"] += 1
        b["by_call_site"][c["call_site"]] = b["by_call_site"].get(c["call_site"], 0) + 1
    synth = [c["provider"] for r in rows for c in r.get("llm_calls", []) if isinstance(c, dict)
             and c.get("call_site") == "synthesize" and c.get("status") == "ok"]
    return {"llm_calls_ok": len(calls), "by_provider": by,
            "synthesis_provider": {p: synth.count(p) for p in sorted(set(synth))},
            "requests_using_both": sum(1 for r in rows if len(r.get("providers_used") or []) > 1), "requests": len(rows)}


def ragas(d):
    p = P9 / d / "ragas.json"
    return json.loads(p.read_text())["summary"] if p.exists() else None


def retrieval_diag(rows):
    rag = [r for r in rows if TESTS[r["id"]]["ragas"]]
    ranks = [v for r in rag for v in (r.get("reference_ranks") or {}).values()]
    return {"items": len(rag),
            "item_any_reference_chunk_retrieved": sum(1 for r in rag if r.get("reference_chunk_hit")),
            "item_reference_paper_retrieved": sum(1 for r in rag if r.get("reference_doc_hit")),
            "reference_chunks": len(ranks), "reference_chunks_retrieved": sum(1 for v in ranks if v),
            "rank_of_retrieved_median": statistics.median([v for v in ranks if v]) if any(ranks) else None}


def main() -> None:
    main_rows = results("routed-deployed")
    multi_rows = [r for r in main_rows if r["category"] == "multi"] + results("routed-deployed-multi")
    simple_rows = [r for r in main_rows if r.get("latency_category") == "simple"]
    out = {
        "configurations": {
            "routed-deployed": "deployed rev paper-analyst-00018-gll (image 74965a1), normal routing, warm, 1 run per item",
            "forced-openai-local": "local uvicorn, code 85e6649, ROUTER_FORCE_PROVIDER=openai (all calls gpt-6.1-sol)",
            "forced-gemini-local": "local uvicorn, code 85e6649, ROUTER_FORCE_PROVIDER=gemini-openrouter (all calls google/gemini-3.8-flash)",
        },
        "latency_warm_deployed": {
            "simple_all": lat_block(simple_rows),
            "simple_rag_items": lat_block([r for r in simple_rows if TESTS[r["id"]]["ragas"]]),
            "simple_data_items": lat_block([r for r in simple_rows if r["category"] == "data"]),
            "multi": lat_block(multi_rows),
            "system_items": lat_block([r for r in main_rows if r["category"] == "system"]),
        },
        "cold_start": {d: (json.loads((P9 / d / "results.json").read_text())) for d in ("cold-start-1", "cold-start-2")
                       if (P9 / d / "results.json").exists()},
        "routing_deployed": provider_share(main_rows + results("routed-deployed-multi")),
        "retrieval_diagnostic": {d: retrieval_diag(results(d)) for d in ("routed-deployed", "forced-openai-local", "forced-gemini-local")},
        "ragas": {d: ragas(d) for d in ("routed-deployed", "forced-openai-local", "forced-gemini-local")},
        "provider_comparison": {d: lat_block([r for r in results(d) if TESTS[r["id"]]["ragas"]])
                                for d in ("routed-deployed", "forced-openai-local", "forced-gemini-local")},
        "system_checks": json.loads((P9 / "routed-deployed/system_checks.json").read_text())["summary"]
        if (P9 / "routed-deployed/system_checks.json").exists() else None,
    }
    for d in ("routed-deployed", "forced-openai-local", "forced-gemini-local"):
        if out["ragas"][d]:
            out["ragas"][d].pop("judge_usage", None)
    (P9 / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("latency_warm_deployed", "routing_deployed", "retrieval_diagnostic", "provider_comparison")}, indent=1))


if __name__ == "__main__":
    main()
