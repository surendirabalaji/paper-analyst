"""Run questions against a running app and summarise routing, per-call latency and worker spans.

Usage: python scripts/measure_routing.py BASE_URL OUT_DIR "question 1" ["question 2" ...]

For each question it writes OUT_DIR/<n>.events.jsonl (every SSE event) and prints one summary block:
provider per call, latency, tokens, worker span vs WORKER_TIMEOUT_S, TTFT and total. It reads no
credentials; it only talks to the app's public /api/chat endpoint.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys
import time

import httpx


async def run(client: httpx.AsyncClient, base: str, query: str) -> list[dict]:
    t0 = time.perf_counter()
    buf, events = "", []
    async with client.stream("POST", f"{base}/api/chat", json={"query": query}) as r:
        r.raise_for_status()
        async for text in r.aiter_text():
            buf += text
            *frames, buf = buf.split("\n\n")
            for f in frames:
                lines = [ln for ln in f.split("\n") if ln and not ln.startswith(":")]
                if len(lines) < 2:
                    continue
                ev = {"event": lines[0][7:], "data": json.loads(lines[1][6:]),
                      "client_ms": int((time.perf_counter() - t0) * 1000)}
                events.append(ev)
    return events


def summarise(query: str, events: list[dict]) -> dict:
    routes, results, fallbacks, spans, open_spans = {}, [], [], [], {}
    done = {}
    first_token_client = None
    for e in events:
        d, name = e["data"], e["event"]
        if name == "llm_route":
            routes[d["decision_id"]] = d
        elif name == "llm_result":
            results.append(d)
        elif name == "fallback":
            fallbacks.append(d)
        elif name == "transition" and d.get("from") == "supervisor" and d.get("task_id"):
            open_spans[d["task_id"]] = d["t_ms"]
        elif name == "worker_result" and d.get("task_id") in open_spans:
            spans.append((d["task_id"], d["agent"], d["status"], d["t_ms"] - open_spans.pop(d["task_id"])))
        elif name == "token" and first_token_client is None:
            first_token_client = e["client_ms"]
        elif name == "done":
            done = d
    calls = []
    for r in results:
        rt = routes.get(r.get("decision_id"), {})
        calls.append({
            "call_site": r.get("call_site"), "agent": r.get("agent"), "provider": r.get("provider"),
            "model": rt.get("model_id"), "dimension": rt.get("dimension"), "status": r.get("status", "ok"),
            "latency_ms": r.get("latency_ms"), "in": r.get("input_tokens"), "out": r.get("output_tokens"),
            "reasoning": r.get("reasoning_tokens"), "upstream": r.get("upstream_provider"),
            "cost_usd": r.get("cost_usd", r.get("est_cost_usd")), "cost_source": r.get("cost_source"),
            "est_cost_usd": r.get("est_cost_usd"), "reported_cost_usd": r.get("reported_cost_usd"),
            "chunks": r.get("stream_chunks"),
        })
    return {
        "query": query,
        "status": done.get("status"),
        "timings": done.get("timings"),
        "client_first_token_ms": first_token_client,
        "worker_spans": spans,
        "calls": calls,
        "fallbacks": fallbacks,
        "providers_used": sorted({c["provider"] for c in calls if c["status"] == "ok"}),
        "cost_usd": round(sum(c["cost_usd"] or 0 for c in calls), 5),
        "n_citations": sum(1 for e in events if e["event"] == "citation"),
        "n_tokens": sum(1 for e in events if e["event"] == "token"),
    }


async def main() -> None:
    base, out = sys.argv[1].rstrip("/"), pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=60)) as client:
        for i, q in enumerate(sys.argv[3:], 1):
            events = await run(client, base, q)
            (out / f"{i}.events.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n")
            s = summarise(q, events)
            summaries.append(s)
            print(f"\n=== {q}\n  status={s['status']} timings={s['timings']} citations={s['n_citations']} "
                  f"token_events={s['n_tokens']} providers={s['providers_used']} cost=${s['cost_usd']}")
            for span in s["worker_spans"]:
                print(f"  worker {span[0]} {span[1]} {span[2]}: {span[3]} ms")
            for c in s["calls"]:
                print(f"  {c['agent']:<12} {c['call_site']:<17} {c['provider']:<18} {c['status']:<9} "
                      f"{c['latency_ms']:>6} ms in={c['in']} out={c['out']} reasoning={c['reasoning']} "
                      f"upstream={c['upstream']} chunks={c['chunks']} dim={c['dimension']} "
                      f"cost={c['cost_usd']} ({c['cost_source']}; est={c['est_cost_usd']} reported={c['reported_cost_usd']})")
            for f in s["fallbacks"]:
                print(f"  FALLBACK {f}")
    (out / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
