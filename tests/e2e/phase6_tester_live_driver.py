"""Tester Phase 6 live driver (REAL providers / Tavily / SQLite through a running app, e.g. the deployed URL).

Usage: python tests/e2e/phase6_tester_live_driver.py BASE_URL OUT_DIR CASES_JSON [--concurrent]
CASES_JSON: [{"id": "...", "query": "..."}]
Writes OUT_DIR/<id>.events.jsonl (every SSE event with client arrival ms) and OUT_DIR/<id>.summary.json.
Collects: plan, transitions, handoff events, every sql_result (status, sql, rows, error kind), tool calls,
tavily_search calls (agent), per-call routing (agent, call site, provider, model returned, dimension,
features, rationale), citations by prefix, C# resolution via /api/source, D# citation payloads, answer,
timings, cost, and a credential-shape scan (booleans only). Reads no credentials.
"""
import asyncio
import json
import pathlib
import re
import sys
import time

import httpx

KEY_SHAPES = [re.compile(r"tavilyApiKey(=|%3D)(?!REDACTED)[^&\s'\"]+", re.I),
              re.compile(r"\btvly-(?!REDACTED)[A-Za-z0-9_-]{6,}"), re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")]


async def run_case(client, base, case, out):
    t0 = time.perf_counter()
    buf, raw, events = "", [], []
    async with client.stream("POST", f"{base}/api/chat", json={"query": case["query"]}) as r:
        code, ctype = r.status_code, r.headers.get("content-type")
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
    (out / f"{case['id']}.events.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n")
    ev = lambda n: [e["data"] for e in events if e["event"] == n]  # noqa: E731
    cites = ev("citation")
    c_res = []
    for c in cites:
        if c["type"] == "paper_chunk":
            rr = await client.get(base + c["href"])
            c_res.append({"cite_id": c["cite_id"], "status": rr.status_code})
    tokens = [e for e in events if e["event"] == "token"]
    done = (ev("done") or [{}])[0]
    rid = {e["data"].get("request_id") for e in events}
    summary = {
        "id": case["id"], "query": case["query"], "http": code, "content_type": ctype,
        "request_ids": sorted(x for x in rid if x),
        "plan": [(t.get("worker"), t.get("depends_on"), t.get("needs_live")) for t in (ev("plan") or [{}])[0].get("tasks", [])],
        "transitions": [(d.get("from"), d.get("to"), d.get("reason")) for d in ev("transition")],
        "handoffs": ev("handoff"),
        "sql": [{k: d.get(k) for k in ("agent", "status", "error_kind", "sql", "n_rows", "rows", "truncated",
                                        "cite_ids", "repair_rounds_left", "message_safe")} for d in ev("sql_result")],
        "tool_calls": [(d.get("agent"), d.get("tool"), d.get("args_safe")) for d in ev("tool_call")],
        "tool_results": [(d.get("agent"), d.get("tool"), d.get("status"), d.get("error_kind")) for d in ev("tool_result")],
        "web_search": [(d.get("agent"), d.get("status"), d.get("query")) for d in ev("web_search")],
        "routes": [{k: d.get(k) for k in ("call_site", "agent", "provider", "model_id", "dimension", "features", "rationale", "fallback_of")}
                   for d in ev("llm_route")],
        "llm_calls": [{k: d.get(k) for k in ("call_site", "agent", "provider", "model_id", "model_returned", "upstream_provider", "status", "attempt", "error_kind", "cost_usd")} for d in ev("llm_result")],
        "worker_results": [(d.get("task_id"), d.get("agent"), d.get("status"), d.get("error_kind")) for d in ev("worker_result")],
        "citations": [{k: c.get(k) for k in ("cite_id", "type", "href", "found_by", "sql", "rows")} for c in cites],
        "c_resolution": c_res,
        "answer": "".join(d.get("delta", "") for d in ev("token")),
        "n_tokens": len(tokens), "first_token_client_ms": tokens[0]["client_ms"] if tokens else None,
        "done": done, "errors": ev("error"),
        "total_client_ms": int((time.perf_counter() - t0) * 1000),
        "key_shape_hits": sum(bool(p.search("".join(raw))) for p in KEY_SHAPES),
    }
    (out / f"{case['id']}.summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    return summary


async def main():
    base, out, cases = sys.argv[1].rstrip("/"), pathlib.Path(sys.argv[2]), json.loads(pathlib.Path(sys.argv[3]).read_text())
    out.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=60)) as client:
        if "--concurrent" in sys.argv:
            res = await asyncio.gather(*(run_case(client, base, c, out) for c in cases))
        else:
            res = [await run_case(client, base, c, out) for c in cases]
    for s in res:
        cost = sum(c.get("cost_usd") or 0 for c in s["llm_calls"])
        print(s["id"], s["http"], s["done"].get("status"), "plan", s["plan"], "sql", [(x["status"], x["error_kind"], x["n_rows"]) for x in s["sql"]],
              "cites", [c["cite_id"] for c in s["citations"]], f"cost={cost:.4f}", "keys", s["key_shape_hits"])


if __name__ == "__main__":
    asyncio.run(main())
