"""Tester Phase 8b: REAL deployed service (REAL providers, Tavily Remote MCP, papers.db, Chroma). Held-out questions.

For each request it records every SSE event with its client arrival time and checks the Phase 8b contract and
the earlier-phase behaviour it could regress:
  - TTFT (client, first `token` event) and total; the server's done.timings
  - speculative_retrieval kept  <=>  the plan is exactly one Research task; kept search used as a seeded call
  - seeded tool_call events match the plan event's seeds; evidence_only  <=>  one Research/Summary task
  - every llm_route has provider/model/dimension/rationale and a matching llm_result with model_returned
  - every [C#]/[W#]/[D#] marker in the answer has a citation event; C# hrefs resolve on the server to the same
    chunk id / doc id; W# have retrieved_at; D# carry SQL and rows; done.unresolved_citations is empty
  - incremental streaming: >1 token events spread over time before `done`
Case-specific checks (Data -> Summary hand-off truth from a local read-only SQL run, MCP sharing, out-of-scope
abstention, faults through the guarded header, concurrency isolation) are evaluated in `case_checks`.

Usage: python tests/e2e/phase8b_tester_deployed.py BASE_URL OUT_DIR [--only KEY ...]
Reads FAULT_INJECT_TOKEN from the environment for the fault cases (never printed or saved).
"""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import re
import sqlite3
import statistics
import sys
import time
import urllib.parse
from datetime import datetime, timezone

import httpx

ROOT = pathlib.Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "papers.db"

SIMPLE = [
    ("h_reflexion_memory", "Where does a Reflexion agent keep its self-reflections between trials, and how are they used?"),
    ("h_toolformer_filter", "What criterion does Toolformer use to decide whether an inserted API call is worth keeping?"),
    ("h_lost_middle", "Why does accuracy drop when the relevant passage sits in the middle of a long context?"),
    ("h_memory_recall", "How do LLM agents decide which past experiences to recall once their memory gets large?"),
    ("h_code_actions", "Why would executable Python be a better action format than JSON tool calls for an agent?"),
    ("h_voyager_skills", "Explain how Voyager's skill library grows over time."),
]
MULTI = [
    ("m_mem2026_handoff", "Which corpus papers from 2026 have 'memory' in their title? Briefly summarize the newest one."),
    ("m_mcp_share", "What are the latest developments around the Model Context Protocol this year, and has the paper "
                    "'Model Context Protocol (MCP): Landscape, Security Threats, and Future Research Directions' "
                    "been published at a venue yet?"),
]
EDGE = [
    ("e_out_of_scope", "What is the recommended daily vitamin D intake for adults?"),
    ("e_missing_paper", "Summarize the corpus paper 'Quantum Swarm Agents for Protein Folding'."),
]
FAULTS = [  # (key, query, header spec)
    ("f_corpus_tool_error", "How do retrieval-augmented systems cope with irrelevant retrieved passages?",
     "tool:corpus_search:error"),
    ("f_gemini_down_summary", "What environments does the AgentBench paper use?", "provider:gemini-openrouter:down"),
]
CONCURRENT = [
    ("c_a", "How does MetaGPT assign roles to its agents?"),
    ("c_b", "What does the RAPTOR paper build its retrieval tree from?"),
]
MARK = re.compile(r"\[((?:[CWD]\d+)(?:\s*,\s*[CWD]\d+)*)\]")


async def run_one(client: httpx.AsyncClient, base: str, key: str, query: str, out: pathlib.Path,
                  headers: dict | None = None) -> dict:
    events: list[dict] = []
    first_tok = None
    t0 = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    async with client.stream("POST", f"{base}/api/chat", json={"query": query}, headers=headers or {}) as r:
        status = r.status_code
        ev = None
        buf = ""
        async for chunk in r.aiter_text():
            buf += chunk
            while "\n\n" in buf:
                frame, buf = buf.split("\n\n", 1)
                now = time.perf_counter() - t0
                m = re.search(r"^event: (.*)$", frame, re.M)
                d = re.search(r"^data: (.*)$", frame, re.M)
                if m and d:
                    ev = m.group(1).strip()
                    data = json.loads(d.group(1))
                    events.append({"event": ev, "client_s": round(now, 3), "data": data})
                    if ev == "token" and first_tok is None:
                        first_tok = now
    total = time.perf_counter() - t0
    (out / f"{key}.run1.events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return {"key": key, "query": query, "raw": f"{key}.run1.events.jsonl", "http_status": status,
            "ttft_s": round(first_tok, 3) if first_tok else None, "total_s": round(total, 3),
            "started_utc": started, "events": events}


def E(res, name):
    return [e["data"] for e in res["events"] if e["event"] == name]


async def generic_checks(client, base, res) -> dict:
    c: dict = {}
    done = (E(res, "done") or [{}])[0]
    plan = (E(res, "plan") or [{}])[0]
    tasks = plan.get("tasks", [])
    workers = [t["worker"] for t in tasks]
    c["status"] = done.get("status")
    c["workers"] = workers
    c["ttft_s"], c["total_s"] = res["ttft_s"], res["total_s"]
    c["server_timings"] = done.get("timings")
    # speculative contract
    spec = E(res, "speculative_retrieval")
    single_research = len(tasks) == 1 and workers[0] == "research"
    c["spec"] = spec[0]["status"] if spec else None
    c["spec_consistent"] = (not spec) or ((spec[0]["status"] == "kept") == single_research)
    seeded = [e for e in E(res, "tool_call") if e.get("seeded_by")]
    plan_seeds = [(s["tool"], s.get("by")) for t in tasks for s in t.get("seeds", [])]
    c["plan_seeds"] = plan_seeds
    c["seeded_calls"] = [(e["tool"], e["seeded_by"]) for e in seeded]
    c["seeds_match_plan"] = sorted(plan_seeds) == sorted(c["seeded_calls"]) or done.get("status") != "done"
    if c["spec"] == "kept":
        c["kept_spec_used"] = any(r.get("speculative") for r in E(res, "retrieval"))
    eo = [w.get("evidence_only", False) for w in E(res, "worker_result")]
    c["evidence_only"] = eo
    want_eo = len(tasks) == 1 and workers[0] in ("research", "summary")
    c["evidence_only_consistent"] = all(x == want_eo for x in eo)
    # routing truthfulness
    routes, results = E(res, "llm_route"), E(res, "llm_result")
    by_id = {}
    for r in results:
        by_id.setdefault(r["decision_id"], []).append(r)
    fb_ids = {f.get("decision_id") for f in E(res, "fallback")}
    c["routes"] = [(r["call_site"], r["agent"], r["provider"], r["dimension"], r.get("reasoning_effort")) for r in routes]
    c["route_fields_ok"] = all(r.get("provider") and r.get("model_id") and r.get("rationale") and r.get("dimension")
                               for r in routes)
    ok_results = [x for x in results if x["status"] == "ok"]
    c["model_returned_matches"] = all(x.get("model_returned", "").startswith(x["model_id"].split("/")[-1].split("-20")[0])
                                      or x["model_id"] in x.get("model_returned", "") or
                                      x.get("model_returned", "").startswith(x["model_id"]) for x in ok_results)
    c["model_pairs"] = sorted({(x["model_id"], x.get("model_returned")) for x in ok_results})
    c["providers_used"] = done.get("providers_used")
    c["synth_provider"] = [r["provider"] for r in routes if r["call_site"].startswith("synth")]
    c["unmatched_routes"] = [r["decision_id"] for r in routes if r["decision_id"] not in by_id and r["decision_id"] not in fb_ids]
    # streaming
    toks = [e for e in res["events"] if e["event"] == "token"]
    done_t = next((e["client_s"] for e in res["events"] if e["event"] == "done"), None)
    c["n_token_events"] = len(toks)
    c["token_spread_s"] = round(toks[-1]["client_s"] - toks[0]["client_s"], 3) if len(toks) > 1 else 0
    c["tokens_before_done"] = all(t["client_s"] <= done_t for t in toks) if done_t else False
    # citations
    answer = "".join(t["data"]["delta"] for t in toks)
    c["answer_chars"] = len(answer)
    marked = {x.strip() for m in MARK.findall(answer) for x in m.split(",")}
    cites = {d["cite_id"]: d for d in E(res, "citation")}
    c["marked"] = sorted(marked, key=lambda s: (s[0], int(s[1:])))
    c["marked_without_citation_event"] = sorted(marked - set(cites))
    c["unresolved_citations"] = done.get("unresolved_citations")
    c["dropped_citations"] = done.get("dropped_citations")
    bad = []
    for cid, d in cites.items():
        if d["type"] == "paper_chunk":
            r = await client.get(base + d["href"])
            j = r.json() if r.status_code == 200 else {}
            if r.status_code != 200 or j.get("chunk_id") != d["chunk_id"] or j.get("doc_id") != d["doc_id"] or not j.get("text"):
                bad.append((cid, r.status_code))
        elif d["type"] == "web":
            if not (d.get("retrieved_at") and d.get("source_url", "").startswith("http")):
                bad.append((cid, "web-missing-provenance"))
        else:
            if d.get("rows") is None and not d.get("sql"):
                bad.append((cid, "data-missing-sql"))
    c["citation_resolution_failures"] = bad
    c["n_citations"] = len(cites)
    c["cost_usd"] = done.get("cost_usd")
    c["errors"] = [(e.get("error_kind"), e.get("message", "")[:120]) for e in E(res, "error")]
    c["answer_head"] = answer[:400]
    c["answer"] = answer
    return c


def sql_truth_mem2026() -> list[tuple[str, str]]:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = con.execute("SELECT arxiv_id, title FROM papers WHERE title LIKE '%memory%' AND published >= '2026-01-01' "
                       "AND published < '2027-01-01' ORDER BY published DESC").fetchall()
    con.close()
    return rows


def case_checks(key: str, res: dict, g: dict) -> dict:
    c: dict = {}
    if key == "m_mem2026_handoff":
        truth = sql_truth_mem2026()
        c["sql_truth"] = truth
        hand = E(res, "handoff")
        c["handoff"] = hand
        handed_ids = {str(p.get("arxiv_id") if isinstance(p, dict) else p).split("v")[0]
                      for h in hand for p in (h.get("papers") or [])}
        c["handed_ids"] = sorted(handed_ids)
        newest = truth[0][0] if truth else None
        summary_reads = [e["args_safe"] for e in E(res, "tool_call") if e["agent"] == "summary"]
        c["summary_reads"] = summary_reads
        c["summary_read_newest"] = any(newest in json.dumps(a) for a in summary_reads)
        c["answer_mentions_all_truth"] = all(t[1].split(":")[0][:25].lower() in g["answer"].lower() or t[0] in g["answer"]
                                             for t in truth)
        c["has_D_and_C"] = any(x.startswith("D") for x in g["marked"]) and any(x.startswith("C") for x in g["marked"])
    if key == "m_mcp_share":
        calls = [(e["agent"], e["args_safe"]) for e in E(res, "tool_call") if e["tool"] == "tavily_search"]
        c["tavily_calls"] = calls
        c["agents_calling_tavily"] = sorted({a for a, _ in calls})
        c["W_marked"] = [x for x in g["marked"] if x.startswith("W")]
    if key == "e_out_of_scope":
        c["C_marked"] = [x for x in g["marked"] if x.startswith("C")]
        c["W_marked"] = [x for x in g["marked"] if x.startswith("W")]
    if key == "e_missing_paper":
        c["C_marked"] = [x for x in g["marked"] if x.startswith("C")]
        c["notes"] = (E(res, "plan") or [{}])[0].get("notes")
        c["task_status"] = [(t["id"], t["status"], t.get("error_kind")) for t in (E(res, "done") or [{}])[0].get("tasks", [])]
    if key.startswith("f_"):
        c["faults_injected"] = (E(res, "done") or [{}])[0].get("faults_injected")
        c["fallbacks"] = [(f.get("from_provider"), f.get("to_provider"), f.get("error_kind")) for f in E(res, "fallback")]
        c["task_status"] = [(t["id"], t["status"], t.get("error_kind")) for t in (E(res, "done") or [{}])[0].get("tasks", [])]
        c["answer_has_injected_label"] = "injected test fault" in g["answer"]
    return c


async def main() -> int:
    base, out = sys.argv[1].rstrip("/"), pathlib.Path(sys.argv[2])
    only = sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else None
    out.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("FAULT_INJECT_TOKEN", "")
    report = {"base": base, "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "runs": []}
    async with httpx.AsyncClient(timeout=httpx.Timeout(200, connect=20)) as client:
        h = await client.get(base + "/health")
        report["health"] = h.json()
        # warm-up (unrecorded in medians)
        await run_one(client, base, "warmup", "What is ReAct?", out)
        todo = [(k, q, None, "simple") for k, q in SIMPLE] + [(k, q, None, "multi") for k, q in MULTI] + \
               [(k, q, None, "edge") for k, q in EDGE]
        if token:
            todo += [(k, q, {"X-Fault-Inject": spec, "X-Fault-Token": token}, "fault") for k, q, spec in FAULTS]
        for key, q, hdr, cat in todo:
            if only and key not in only:
                continue
            res = await run_one(client, base, key, q, out, hdr)
            g = await generic_checks(client, base, res)
            row = {"key": key, "category": cat, "query": q, "raw": res["raw"], "started_utc": res["started_utc"],
                   "http_status": res["http_status"], **{k: v for k, v in g.items() if k != "answer"},
                   "case": case_checks(key, res, g)}
            report["runs"].append(row)
            print(json.dumps({k: row[k] for k in ("key", "status", "workers", "ttft_s", "total_s", "spec",
                                                  "synth_provider", "n_citations", "cost_usd")}), flush=True)
        if not only or any(k in only for k, _ in CONCURRENT):
            pair = await asyncio.gather(*(run_one(client, base, k, q, out) for k, q in CONCURRENT))
            rids = []
            for res, (k, q) in zip(pair, CONCURRENT):
                g = await generic_checks(client, base, res)
                rid = {e["data"].get("request_id") for e in res["events"]}
                rids.append(rid)
                other_q = [qq for kk, qq in CONCURRENT if kk != k][0]
                foreign_docs = {d["doc_id"] for d in E(res, "citation")}
                row = {"key": k, "category": "concurrent", "query": q, "raw": res["raw"], "started_utc": res["started_utc"],
                       "http_status": res["http_status"], **{kk: v for kk, v in g.items() if kk != "answer"},
                       "case": {"request_ids": sorted(rid), "cited_docs": sorted(foreign_docs),
                                "spec_query_is_own": all(s["query"] == q for s in E(res, "speculative_retrieval")),
                                "other_query_in_events": other_q in json.dumps(res["events"])}}
                report["runs"].append(row)
                print(json.dumps({x: row[x] for x in ("key", "status", "workers", "ttft_s", "spec", "n_citations")}), flush=True)
            report["concurrent_request_ids_disjoint"] = not (rids[0] & rids[1])
    simple = [r["ttft_s"] for r in report["runs"] if r["category"] == "simple" and r["ttft_s"]]
    if simple:
        report["simple_ttft"] = {"n": len(simple), "median": statistics.median(simple), "min": min(simple), "max": max(simple)}
    report["results"] = [{"raw": r["raw"], "query": r["query"]} for r in report["runs"]]  # for faithfulness_check.py
    (out / "results.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"simple_ttft": report.get("simple_ttft"),
                      "total_cost": round(sum(r["cost_usd"] or 0 for r in report["runs"]), 4)}))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
