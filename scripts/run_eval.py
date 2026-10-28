#!/usr/bin/env python3
"""Phase 9 evaluation runner: stream frozen test-set questions against a URL and record everything needed
for RAGAS, latency, cost and provider analysis. The system gets the question only (end to end); reference
answers and chunk ids are never sent.

Per item it saves the raw SSE events (`<id>.run<n>.events.jsonl`) and a row in `results.json`:
client TTFT/total, server timings, plan, every LLM call (agent, call site, provider, model, tokens, latency,
cost), the retrieved chunk ids in first-seen order, cited ids, and the rank of each reference chunk in the
retrieved list (retrieval diagnostic, no LLM). The forced-provider runs point this at a local server started
with ROUTER_FORCE_PROVIDER; the label is read back from the routing events, not assumed.

Usage: .venv/bin/python scripts/run_eval.py URL OUT_DIR [--ids s01 c02 ...] [--categories rag_single ...]
                                           [--runs 1] [--label routed-deployed]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from measure_latency import run_one  # noqa: E402  (same client-side timing as the Phase 8b measurements)

TESTSET = ROOT / "eval/testset/testset_v1.jsonl"
FROZEN_SHA256 = "50e9c0f3e629e9bc39f6412e72133ab8c925b61f4eeaf22e791680cf1f8f5382"


def load_events(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def analyse(item: dict, events: list[dict]) -> dict:
    retrieved: list[dict] = []
    seen = set()
    for e in events:
        if e["event"] != "retrieval":
            continue
        for c in e["data"].get("chunks", []):
            key = c.get("chunk_id") or c.get("source_url") or c.get("cite_id")
            if key and key not in seen:
                seen.add(key)
                retrieved.append({"key": key, "type": c.get("type"), "agent": e["data"].get("agent"),
                                  "tool": e["data"].get("tool", "corpus_search")})
    chunk_order = [r["key"] for r in retrieved if r["type"] == "paper_chunk"]
    refs = item.get("reference_chunk_ids", [])
    ranks = {c: (chunk_order.index(c) + 1 if c in chunk_order else None) for c in refs}
    ref_docs = {c.split("::")[0] for c in refs}
    calls = [{k: d.get(k) for k in ("agent", "call_site", "provider", "model_id", "model_returned", "status",
                                     "error_kind", "attempt", "latency_ms", "ttft_ms", "input_tokens",
                                     "output_tokens", "reasoning_tokens", "cached_input_tokens", "cost_usd",
                                     "cost_source")}
             for d in (e["data"] for e in events if e["event"] == "llm_result")]
    routes = [{k: d.get(k) for k in ("agent", "call_site", "provider", "dimension", "rationale")}
              for d in (e["data"] for e in events if e["event"] == "llm_route")]
    tools = [{"agent": d.get("agent"), "tool": d.get("tool"), "status": d.get("status"), "seeded_by": d.get("seeded_by")}
             for d in (e["data"] for e in events if e["event"] == "tool_result")]
    return {
        "retrieved": retrieved,
        "n_retrieved_chunks": len(chunk_order),
        "reference_ranks": ranks,
        "reference_chunk_hit": any(v is not None for v in ranks.values()) if refs else None,
        "reference_doc_hit": bool(ref_docs & {k.split("::")[0] for k in chunk_order}) if refs else None,
        "cited": [e["data"].get("cite_id") for e in events if e["event"] == "citation"],
        "llm_calls": calls,
        "routes": routes,
        "tool_results": tools,
        "forced": any(r["dimension"] == "forced" for r in routes),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out_dir")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--categories", nargs="*")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    raw = TESTSET.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != FROZEN_SHA256:
        sys.exit(f"test set hash {sha} does not match the frozen v1 hash; refusing to run")
    items = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    if a.ids:
        items = [i for i in items if i["id"] in a.ids]
    if a.categories:
        items = [i for i in items if i["category"] in a.categories]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.json"
    prev = json.loads(res_path.read_text()) if res_path.exists() else {"results": []}
    results = prev["results"]
    url = a.url.rstrip("/")
    with httpx.Client(timeout=httpx.Timeout(330, connect=30)) as client:
        for r in range(1, a.runs + 1):
            for it in items:
                raw_path = out / f"{it['id']}.run{r}.events.jsonl"
                started = datetime.now(timezone.utc).isoformat(timespec="seconds")
                t0 = time.perf_counter()
                try:
                    row = run_one(client, url, it["question"], raw_path)
                except Exception as e:  # recorded, never retried silently
                    row = {"status": "client_error", "errors": [f"{type(e).__name__}: {str(e)[:200]}"],
                           "total_s": round(time.perf_counter() - t0, 3), "raw": raw_path.name}
                if raw_path.exists():
                    row.update(analyse(it, load_events(raw_path)))
                row.update({"id": it["id"], "run": r, "category": it["category"],
                            "latency_category": it.get("latency_category"), "query": it["question"],
                            "started_utc": started})
                results = [x for x in results if not (x["id"] == it["id"] and x["run"] == r)] + [row]
                print(f"{it['id']} run{r}: {row.get('status')} ttft={row.get('ttft_s')} total={row.get('total_s')} "
                      f"workers={row.get('workers')} providers={row.get('providers_used')} cost={row.get('cost_usd')} "
                      f"ref_rank={row.get('reference_ranks')}", flush=True)
                res_path.write_text(json.dumps({"url": url, "label": a.label, "testset_sha256": sha,
                                                "results": results}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
