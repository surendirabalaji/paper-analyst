"""Tester Phase 5 live driver (REAL providers, REAL Tavily Remote MCP, through a running app).

Usage: python tests/e2e/phase5_tester_live_driver.py BASE_URL OUT_DIR CASES_JSON [--concurrent]
CASES_JSON: [{"id": "...", "query": "..."}]
Writes OUT_DIR/<id>.events.jsonl (every SSE event + client arrival ms) and OUT_DIR/<id>.summary.json.

Collects: plan (workers, needs_live), tool_gate decisions, every tavily_search call (agent, purpose, query,
status, W# ids), per-call routing (call site, agent, provider, model returned, dimension, rationale),
citations split C#/W#, C# resolution through /api/source, W# provenance (URL in an MCP result of this
request), dropped/unresolved markers, timings, cost and a credential-shape scan of the raw stream.
No credentials are read or written by this script; the scan only reports booleans/counts.
"""
import asyncio
import json
import pathlib
import re
import sys
import time

import httpx

KEY_SHAPES = [
    re.compile(r"tavilyApiKey(=|%3D)(?!REDACTED)[^&\s'\"]+", re.I),
    re.compile(r"\btvly-(?!REDACTED)[A-Za-z0-9_-]{6,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
]


async def run_case(client, base, case, out):
    t0 = time.perf_counter()
    buf, raw, events = "", [], []
    async with client.stream("POST", f"{base}/api/chat", json={"query": case["query"]}) as r:
        status_code, ctype = r.status_code, r.headers.get("content-type")
        async for text in r.aiter_text():
            raw.append(text)
            buf += text
            *done, buf = buf.split("\n\n")
            now = int((time.perf_counter() - t0) * 1000)
            for f in done:
                lines = f.split("\n")
                if lines[0].startswith(":"):
                    continue
                assert lines[0].startswith("event: ") and lines[1].startswith("data: ") and len(lines) == 2, f
                events.append({"event": lines[0][7:], "client_ms": now, "data": json.loads(lines[1][6:])})
    total_client_ms = int((time.perf_counter() - t0) * 1000)
    (out / f"{case['id']}.events.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n")
    rawtext = "".join(raw)

    ev = lambda name: [e["data"] for e in events if e["event"] == name]  # noqa: E731
    plan = (ev("plan") or [{}])[0]
    done = (ev("done") or [{}])[0]
    answer = "".join(d["delta"] for d in ev("token"))
    webs = ev("web_search")
    web_urls = {r["cite_id"]: r["url"] for w in webs if w["status"] == "ok" for r in w["results"]}
    cites = ev("citation")
    c_res = []
    for c in cites:
        if c["type"] == "paper_chunk":
            rr = await client.get(base + c["href"])
            j = rr.json() if rr.status_code == 200 else {}
            c_res.append({"cite_id": c["cite_id"], "doc_id": c["doc_id"], "http": rr.status_code,
                          "match": j.get("chunk_id") == c["chunk_id"] and j.get("doc_id") == c["doc_id"],
                          "section": j.get("section_path"), "text_head": (j.get("text") or "")[:240]})
        else:
            c_res.append({"cite_id": c["cite_id"], "type": "web", "url": c["source_url"], "found_by": c.get("found_by"),
                          "retrieved_at": c.get("retrieved_at"),
                          "in_mcp_results_of_this_request": web_urls.get(c["cite_id"]) == c["source_url"]})
    tok_client = [e["client_ms"] for e in events if e["event"] == "token"]
    summary = {
        "id": case["id"], "query": case["query"], "http_status": status_code, "content_type": ctype,
        "request_ids": sorted({e["data"]["request_id"] for e in events}),
        "plan": {"source": plan.get("source"), "summary": plan.get("plan_summary"), "tasks": plan.get("tasks")},
        "tool_gates": [{k: g.get(k) for k in ("agent", "enabled", "reason", "best_corpus_score")} for g in ev("tool_gate")],
        "tool_calls": [(d["agent"], d["tool"], d.get("args_safe")) for d in ev("tool_call")],
        "tool_results": [(d["agent"], d["tool"], d["status"], d.get("error_kind"), d.get("n_results")) for d in ev("tool_result")],
        "web_searches": [{"agent": w["agent"], "tool": w["tool"], "server": w["server"], "transport": w["transport"],
                          "purpose": w["purpose"], "query": w["query"], "time_range": w.get("time_range"),
                          "status": w["status"], "error_kind": w.get("error_kind"), "latency_ms": w["latency_ms"],
                          "w_ids": [r["cite_id"] for r in w["results"]],
                          "domains": [r["domain"] for r in w["results"]]} for w in webs],
        "llm_calls": [{"call_site": d["call_site"], "agent": d["agent"], "provider": d["provider"],
                       "model_returned": d.get("model_returned"), "upstream": d.get("upstream_provider"),
                       "status": d["status"], "error_kind": d.get("error_kind"), "latency_ms": d.get("latency_ms"),
                       "in": d.get("input_tokens"), "out": d.get("output_tokens"), "cost_usd": d.get("cost_usd")}
                      for d in ev("llm_result")],
        "routes": [{"call_site": d["call_site"], "agent": d["agent"], "provider": d["provider"], "model_id": d["model_id"],
                    "dimension": d["dimension"], "rationale": d["rationale"],
                    "features": {k: v for k, v in (d.get("features") or {}).items()
                                 if k in ("long_source_tokens_est", "context_tokens_est", "round", "n_tasks_ok",
                                          "n_papers", "n_web", "comparison", "evidence_tokens_est")}}
                   for d in ev("llm_route")],
        "fallbacks": ev("llm_fallback") + ev("fallback"),
        "worker_results": [{k: d.get(k) for k in ("task_id", "agent", "status", "error_kind", "evidence_ids", "rounds", "tool_calls")}
                           for d in ev("worker_result")],
        "citations": c_res,
        "answer_markers": sorted(set(re.findall(r"\b[CW]\d+\b", answer))),
        "done": {k: done.get(k) for k in ("status", "unresolved_citations", "dropped_citations", "timings",
                                           "cost_usd", "cost_basis", "providers_used", "fallbacks")},
        "n_token_events": len(tok_client), "distinct_token_arrivals": len(set(tok_client)),
        "first_token_client_ms": tok_client[0] if tok_client else None, "total_client_ms": total_client_ms,
        "errors": ev("error"),
        "secret_shape_hits_in_stream": sum(len(p.findall(rawtext)) for p in KEY_SHAPES),
        "answer": answer,
    }
    (out / f"{case['id']}.summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


async def main():
    base, out, cases = sys.argv[1].rstrip("/"), pathlib.Path(sys.argv[2]), json.loads(pathlib.Path(sys.argv[3]).read_text())
    out.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
        if "--concurrent" in sys.argv:
            res = await asyncio.gather(*(run_case(client, base, c, out) for c in cases))
        else:
            res = [await run_case(client, base, c, out) for c in cases]
    for s in res:
        print(json.dumps({k: s[k] for k in ("id", "done", "web_searches", "secret_shape_hits_in_stream")}, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    asyncio.run(main())
