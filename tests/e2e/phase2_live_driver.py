"""Tester Phase 2 live driver (REAL OpenAI generation + embeddings through the running app).

Usage: python tests/e2e/phase2_live_driver.py BASE_URL OUT_DIR CASES_JSON [--concurrent]
CASES_JSON: [{"id": "...", "query": "...", "expect_docs": ["2305.18323v1", ...] | [], "expect_abstain": bool}]
Writes OUT_DIR/<id>.events.jsonl (every SSE event + client arrival ms) and OUT_DIR/<id>.summary.json.
No credentials are read or written by this script.
"""
import asyncio
import json
import pathlib
import sys
import time

import httpx


def frames(buf):
    *done, rest = buf.split("\n\n")
    return done, rest


async def run_case(client, base, case, out):
    t0 = time.perf_counter()
    buf, events, reads = "", [], 0
    async with client.stream("POST", f"{base}/api/chat", json={"query": case["query"]}) as r:
        status_code, ctype = r.status_code, r.headers.get("content-type")
        async for text in r.aiter_text():
            reads += 1
            buf += text
            done, buf = frames(buf)
            now = int((time.perf_counter() - t0) * 1000)
            for f in done:
                lines = f.split("\n")
                if lines[0].startswith(":"):
                    events.append({"event": "_comment", "client_ms": now})
                    continue
                assert lines[0].startswith("event: ") and lines[1].startswith("data: ") and len(lines) == 2
                events.append({"event": lines[0][7:], "client_ms": now, "read": reads, "data": json.loads(lines[1][6:])})
    total_client_ms = int((time.perf_counter() - t0) * 1000)
    (out / f"{case['id']}.events.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n")

    toks = [e for e in events if e["event"] == "token"]
    answer = "".join(e["data"]["delta"] for e in toks)
    done = next((e["data"] for e in events if e["event"] == "done"), None)
    cites = [e["data"] for e in events if e["event"] == "citation"]
    retrieved = [c for e in events if e["event"] == "retrieval" for c in e["data"]["chunks"]]
    # resolve every citation through the public source endpoint
    resolved = []
    for c in cites:
        rr = await client.get(base + c["href"])
        j = rr.json() if rr.status_code == 200 else {}
        resolved.append({"cite_id": c["cite_id"], "chunk_id": c["chunk_id"], "http": rr.status_code,
                         "match": j.get("chunk_id") == c["chunk_id"] and j.get("doc_id") == c["doc_id"],
                         "doc_id": c["doc_id"], "section": j.get("section_path"), "text_head": (j.get("text") or "")[:300]})
    tok_server = [e["data"]["t_ms"] for e in toks]
    tok_client = [e["client_ms"] for e in toks]
    summary = {
        "id": case["id"], "query": case["query"], "http_status": status_code, "content_type": ctype,
        "request_ids": sorted({e["data"]["request_id"] for e in events if "data" in e}),
        "event_counts": {k: sum(1 for e in events if e["event"] == k) for k in sorted({e["event"] for e in events})},
        "plan": next((e["data"] for e in events if e["event"] == "plan"), None),
        "transitions": [f'{e["data"]["from"]}->{e["data"]["to"]}: {e["data"]["reason"]}' for e in events if e["event"] == "transition"],
        "tool_calls": [e["data"]["args_safe"] for e in events if e["event"] == "tool_call"],
        "retrieved_docs": sorted({c["doc_id"] for c in retrieved}),
        "llm_calls": [{k: e["data"].get(k) for k in ("agent", "call_site", "provider", "model_returned", "input_tokens", "output_tokens", "latency_ms")}
                      for e in events if e["event"] == "llm_result"],
        "answer": answer,
        "citations": resolved,
        "cited_docs": sorted({c["doc_id"] for c in cites}),
        "done": done,
        "last_event": events[-1]["event"] if events else None,
        "n_token_events": len(toks),
        "n_distinct_reads_with_tokens": len({e["read"] for e in toks}),
        "first_token_client_ms": tok_client[0] if toks else None,
        "last_token_client_ms": tok_client[-1] if toks else None,
        "first_token_server_ms": tok_server[0] if toks else None,
        "last_token_server_ms": tok_server[-1] if toks else None,
        "done_client_ms": next((e["client_ms"] for e in events if e["event"] == "done"), None),
        "total_client_ms": total_client_ms,
        "expect_docs": case.get("expect_docs"),
        "expect_abstain": case.get("expect_abstain", False),
    }
    (out / f"{case['id']}.summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


async def main():
    base, out, cases_path = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
    out.mkdir(parents=True, exist_ok=True)
    cases = json.loads(pathlib.Path(cases_path).read_text())
    async with httpx.AsyncClient(timeout=httpx.Timeout(240, connect=10)) as c:
        if "--concurrent" in sys.argv:
            res = await asyncio.gather(*(run_case(c, base, k, out) for k in cases))
        else:
            res = [await run_case(c, base, k, out) for k in cases]
    for s in res:
        d = s["done"] or {}
        print(json.dumps({"id": s["id"], "status": d.get("status"), "plan_tasks": len((s["plan"] or {}).get("tasks", [])),
                          "tokens": s["n_token_events"], "reads": s["n_distinct_reads_with_tokens"],
                          "cited": s["cited_docs"], "retrieved": s["retrieved_docs"][:6],
                          "resolved": f'{sum(c["match"] for c in s["citations"])}/{len(s["citations"])}',
                          "unresolved": d.get("unresolved_citations"), "ttft": (d.get("timings") or {}).get("ttft_ms"),
                          "total": (d.get("timings") or {}).get("total_ms"), "tok": d.get("tokens_by_provider")}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
