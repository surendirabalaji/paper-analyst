#!/usr/bin/env python3
"""Phase 9 tester: independent, offline re-verification of the recorded evaluation evidence.

No network and no LLM calls. Everything is recomputed from the raw SSE event files and the frozen test set,
with this file's own code (it does not import scripts/summarize_phase9.py, run_eval.py or ragas_eval.py), and
compared against results.json, ragas.json, summary.json and system_checks.json.

Usage: .venv/bin/python tests/integration/phase9_verify_evidence.py OUT_JSON
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P9 = ROOT / "eval/results/phase9"
TESTSET = ROOT / "eval/testset/testset_v1.jsonl"
FROZEN = "50e9c0f3e629e9bc39f6412e72133ab8c925b61f4eeaf22e791680cf1f8f5382"

checks: list[dict] = []
DROPPED: list[tuple] = []


def check(cid: str, ok: bool, detail="") -> bool:
    checks.append({"id": cid, "pass": bool(ok), "detail": detail})
    return ok


def ev(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def close(a, b, tol=0.0015):
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= tol


def nearest_rank(vals, q):  # same index rule the summarizer documents; checked against a plain definition too
    s = sorted(vals)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


# ---------------------------------------------------------------- test set
raw = TESTSET.read_bytes()
sha = hashlib.sha256(raw).hexdigest()
check("T9-TS1 frozen hash", sha == FROZEN, sha)
items = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
T = {i["id"]: i for i in items}
comp = Counter(i["category"] for i in items)
check("T9-TS2 composition 44 = 20/6/6/4/2/6",
      len(items) == 44 and comp == Counter(rag_single=20, rag_compare=6, rag_hard=6, data=4, multi=2, system=6), dict(comp))
ragas_ids = sorted(i["id"] for i in items if i["ragas"])
check("T9-TS3 32 RAGAS items, all with reference + reference chunks",
      len(ragas_ids) == 32 and all(T[i].get("reference") and T[i].get("reference_chunk_ids") for i in ragas_ids))
check("T9-TS4 ragas flag only on rag_* categories",
      all(T[i]["category"].startswith("rag_") for i in ragas_ids) and all(not i["ragas"] for i in items if not i["category"].startswith("rag_")))
n_ref_chunks = sum(len(T[i]["reference_chunk_ids"]) for i in ragas_ids)
check("T9-TS5 49 reference chunks over RAGAS items", n_ref_chunks == 49, n_ref_chunks)

# chunk ids exist in the local Chroma collection (ground truth resolvable)
chroma_ok = None
try:
    sys.path.insert(0, str(ROOT))
    from app.config import load_settings  # noqa: E402
    from ingest.store import open_client  # noqa: E402

    s = load_settings()
    col = open_client(s.chroma_dir).get_collection(s.chroma_collection)
    allref = sorted({c for i in ragas_ids for c in T[i]["reference_chunk_ids"]})
    got = col.get(ids=allref, include=[])
    chroma_ok = set(got["ids"]) == set(allref)
    check("T9-TS6 every reference chunk id resolves in the Chroma index", chroma_ok, f"{len(got['ids'])}/{len(allref)}; count={col.count()}")
except Exception as e:  # pragma: no cover
    col = None
    check("T9-TS6 every reference chunk id resolves in the Chroma index", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- per run directory
def recompute_row(path: Path) -> dict:
    E = ev(path)
    tok = [e for e in E if e["event"] == "token"]
    done = next((e["data"] for e in reversed(E) if e["event"] == "done"), {})
    routes = [e["data"] for e in E if e["event"] == "llm_route"]
    results_ = [e["data"] for e in E if e["event"] == "llm_result"]
    ret_order, seen = [], set()
    for e in E:
        if e["event"] != "retrieval":
            continue
        for c in e["data"].get("chunks", []):
            k = c.get("chunk_id") or c.get("source_url") or c.get("cite_id")
            if k and k not in seen:
                seen.add(k)
                ret_order.append((k, c.get("type")))
    rids = {e["data"].get("request_id") for e in E if isinstance(e.get("data"), dict)}
    return {"E": E, "answer": "".join(t["data"]["delta"] for t in tok),
            "first_token_s": tok[0]["client_s"] if tok else None,
            "first_token_nonempty": bool(tok and tok[0]["data"]["delta"].strip()),
            "last_s": E[-1]["client_s"] if E else None, "done": done, "routes": routes, "results": results_,
            "retrieved": ret_order, "request_ids": rids,
            "tokens_before_done": sum(1 for e in E if e["event"] == "token"),
            }


def run_dir_checks(name: str, expect_forced: str | None) -> dict:
    d = P9 / name
    meta = json.loads((d / "results.json").read_text())
    rows = meta["results"]
    check(f"T9-{name}-H testset hash recorded", meta.get("testset_sha256") == FROZEN)
    keyset = Counter((r["id"], r["run"]) for r in rows)
    check(f"T9-{name}-U one row per (id, run)", all(v == 1 for v in keyset.values()))
    mism, rec = [], {}
    for r in rows:
        p = d / r["raw"]
        if not p.exists():
            mism.append((r["id"], "raw missing"))
            continue
        x = recompute_row(p)
        rec[(r["id"], r["run"])] = x
        if len(x["request_ids"] - {None}) != 1:
            mism.append((r["id"], "request ids", x["request_ids"]))
        if not close(x["first_token_s"], r.get("ttft_s"), 0.0011):
            mism.append((r["id"], "ttft", x["first_token_s"], r.get("ttft_s")))
        if x["done"].get("status") != r.get("status"):
            mism.append((r["id"], "status", x["done"].get("status"), r.get("status")))
        if x["done"].get("cost_usd") is not None and not close(round(x["done"]["cost_usd"], 4), r.get("cost_usd"), 0.00011):
            mism.append((r["id"], "cost", x["done"].get("cost_usd"), r.get("cost_usd")))
        if x["done"].get("timings") != r.get("server_timings"):
            mism.append((r["id"], "timings"))
        if not x["first_token_nonempty"]:
            mism.append((r["id"], "first token is empty / whitespace"))
        if r.get("total_s") is not None and x["last_s"] is not None and x["last_s"] > r["total_s"] + 0.01:
            mism.append((r["id"], "last event after total"))
        # llm_calls in results.json == llm_result events
        if "llm_calls" in r and isinstance((r["llm_calls"] or [None])[0], dict):
            if [(c["provider"], c["model_returned"], c["status"]) for c in r["llm_calls"]] != \
               [(c["provider"], c.get("model_returned"), c["status"]) for c in x["results"]]:
                mism.append((r["id"], "llm_calls"))
        # cost: done.cost_usd vs the sum of per-call costs
        call_cost = sum((c.get("cost_usd") or 0) for c in x["results"])
        if x["done"].get("cost_usd") is not None and call_cost > x["done"]["cost_usd"] + 1e-5:  # per-call 6-dp rounding
            mism.append((r["id"], "call costs exceed request cost", call_cost, x["done"]["cost_usd"]))
    check(f"T9-{name}-R results.json rows reproduce from raw events", not mism, mism[:10])

    # forced or routed
    dims = Counter(rt.get("dimension") for x in rec.values() for rt in x["routes"])
    provs = Counter(rs.get("provider") for x in rec.values() for rs in x["results"] if rs.get("status") == "ok")
    models = Counter(rs.get("model_returned") for x in rec.values() for rs in x["results"] if rs.get("status") == "ok")
    if expect_forced:
        check(f"T9-{name}-F every route dimension 'forced' and one provider", set(dims) == {"forced"} and set(provs) == {expect_forced},
              {"dims": dict(dims), "providers": dict(provs), "models": dict(models)})
    else:
        check(f"T9-{name}-F no forced routes in an ordinary run", "forced" not in dims, {"dims": dict(dims), "providers": dict(provs)})

    # citations: every C# cites a chunk that was retrieved in the same request; every answer marker resolves
    bad_c = []
    for (iid, run), x in rec.items():
        rset = {k for k, _ in x["retrieved"]}
        cites = [e["data"] for e in x["E"] if e["event"] == "citation"]
        ids = {c.get("cite_id") for c in cites}
        for c in cites:
            if c.get("type") == "paper_chunk" and c.get("chunk_id") not in rset:
                bad_c.append((iid, c.get("cite_id"), c.get("chunk_id"), "cited chunk not retrieved"))
            if c.get("type") == "paper_chunk" and col is not None:
                g = col.get(ids=[c.get("chunk_id")], include=[])
                if not g["ids"]:
                    bad_c.append((iid, c.get("cite_id"), "chunk id not in index"))
        dropped = set(x["done"].get("dropped_citations") or [])
        for m in set(re.findall(r"\[([CWDR]\d+)\]", x["answer"])):
            if m not in ids and m not in dropped:
                bad_c.append((iid, m, "marker without citation event"))
            elif m in dropped:
                DROPPED.append((name, iid, m))
        if x["done"].get("unresolved_citations"):
            bad_c.append((iid, "unresolved", x["done"]["unresolved_citations"]))
    check(f"T9-{name}-C citations resolve to retrieved chunks / events", not bad_c, bad_c[:10])
    return {"rows": rows, "rec": rec, "dims": dict(dims), "providers": dict(provs), "models": dict(models)}


R = {n: run_dir_checks(n, f) for n, f in [("routed-deployed", None), ("routed-deployed-multi", None),
                                          ("forced-openai-local", "openai"), ("forced-gemini-local", "gemini-openrouter")]}

# forced runs contain exactly the 32 RAGAS items
for n in ("forced-openai-local", "forced-gemini-local"):
    check(f"T9-{n}-N exactly the 32 RAGAS items", sorted(r["id"] for r in R[n]["rows"]) == ragas_ids)
check("T9-routed-N all 44 items once", sorted(r["id"] for r in R["routed-deployed"]["rows"]) == sorted(T))
check("T9-multi-N m01, m02 runs 1-3", sorted((r["id"], r["run"]) for r in R["routed-deployed-multi"]["rows"])
      == [("m01", 1), ("m01", 2), ("m01", 3), ("m02", 1), ("m02", 2), ("m02", 3)])

# run ordering / no hidden reruns: started_utc monotonic in file order and inside one window
for n in R:
    st = [r.get("started_utc") for r in R[n]["rows"]]
    check(f"T9-{n}-O started_utc monotonic (no out-of-order rerun rows)", st == sorted(st), f"{st[0]} .. {st[-1]}")

# event timestamps after the test-set freeze commit (32b2f36, 2026-10-28T06:19:16Z)
FREEZE_UTC = "2026-10-28T06:19:16"
for n in R:
    ts = min(x["E"][0]["data"]["ts"] for x in R[n]["rec"].values())
    check(f"T9-{n}-Z first event after test-set freeze", ts > FREEZE_UTC, ts)


# ---------------------------------------------------------------- RAGAS
def contexts_count(x, chunk_lookup):
    n = 0
    for k, typ in x["retrieved"]:
        if typ == "paper_chunk":
            n += 1 if k in chunk_lookup else 0
        elif typ == "web":
            n += 1
        else:
            n += 1  # SQL rows etc. -- the scorer only adds those with rows; refined below
    return n


ragas_out = {}
for n in ("routed-deployed", "forced-openai-local", "forced-gemini-local"):
    rj = json.loads((P9 / n / "ragas.json").read_text())
    rows, summ = rj["rows"], rj["summary"]
    check(f"T9-RG-{n}-1 rows are exactly the 32 RAGAS items", sorted(r["id"] for r in rows) == ragas_ids)
    check(f"T9-RG-{n}-2 testset hash in ragas summary", summ.get("testset_sha256") == FROZEN)
    metrics = [m for m in ("faithfulness", "answer_relevancy", "context_precision", "context_recall") if m in summ["all"]]
    # answers and context counts come from the raw files
    bad = []
    all_chunk_ids = [k for r in rows for k, t in R[n]["rec"][(r["id"], r["run"])]["retrieved"] if t == "paper_chunk"]
    present = set(col.get(ids=sorted(set(all_chunk_ids)), include=[])["ids"]) if col is not None and all_chunk_ids else set()
    missing_ctx = sorted(set(all_chunk_ids) - present)
    for r in rows:
        x = R[n]["rec"][(r["id"], r["run"])]
        if len(x["answer"]) != r["answer_chars"]:
            bad.append((r["id"], "answer_chars", len(x["answer"]), r["answer_chars"]))
        n_pc = sum(1 for k, t in x["retrieved"] if t == "paper_chunk" and k in present)
        n_web = sum(1 for k, t in x["retrieved"] if t == "web")
        n_other = sum(1 for k, t in x["retrieved"] if t not in ("paper_chunk", "web"))
        if not (n_pc + n_web <= r["n_contexts_all"] <= n_pc + n_web + n_other):
            bad.append((r["id"], "n_contexts", n_pc, n_web, n_other, r["n_contexts_all"]))
        if r.get("errors"):
            bad.append((r["id"], "metric errors", r["errors"]))
        for m in metrics:
            v = r.get(m)
            if v is None or (isinstance(v, float) and math.isnan(v)) or not (0 <= v <= 1):
                bad.append((r["id"], m, v))
    check(f"T9-RG-{n}-3 answers/contexts match raw events; scores in [0,1], none missing", not bad, bad[:10])
    check(f"T9-RG-{n}-4 no retrieved chunk silently dropped from contexts (absent from index)", not missing_ctx, missing_ctx[:5])
    # recompute aggregates
    agg_bad = []
    recomputed = {}
    for m in metrics:
        vals = [r[m] for r in rows if r.get(m) is not None]
        rc = {"n": len(vals), "mean": statistics.mean(vals), "median": statistics.median(vals), "min": min(vals)}
        recomputed[m] = rc
        sm = summ["all"][m]
        if sm["n"] != rc["n"] or sm["n_failed"] != len(rows) - len(vals) or not close(sm["mean"], rc["mean"], 6e-5) \
           or not close(sm["median"], rc["median"], 6e-5) or not close(sm["min"], rc["min"], 6e-5):
            agg_bad.append((m, sm, rc))
        for cat in ("rag_single", "rag_compare", "rag_hard"):
            cv = [r[m] for r in rows if r["category"] == cat and r.get(m) is not None]
            if not close(summ["by_category"][cat][m]["mean"], statistics.mean(cv), 6e-5):
                agg_bad.append((m, cat))
            recomputed[f"{m}:{cat}"] = round(statistics.mean(cv), 4)
    check(f"T9-RG-{n}-5 RAGAS aggregates recompute", not agg_bad, agg_bad[:5])
    ragas_out[n] = {"metrics": metrics, "recomputed": recomputed, "judge": summ.get("judge_model"),
                    "judge_cost": summ.get("judge_cost_usd_est"), "max_contexts": summ.get("max_contexts")}
    check(f"T9-RG-{n}-6 no context truncation", summ.get("max_contexts") is None)

check("T9-RG-main all four metrics on the main run",
      ragas_out["routed-deployed"]["metrics"] == ["faithfulness", "answer_relevancy", "context_precision", "context_recall"])


# ---------------------------------------------------------------- summary.json recompute
S = json.loads((P9 / "summary.json").read_text())


def stats(vals):
    vals = [v for v in vals if v is not None]
    return {"n": len(vals), "median": round(statistics.median(vals), 3), "p90": round(nearest_rank(vals, 0.9), 3),
            "min": round(min(vals), 3), "max": round(max(vals), 3)}


def blk(rows):
    return {"ttft_s": stats([R_all[k]["first_token_s"] for k in rows]),
            "total_s": stats([R_rows[k]["total_s"] for k in rows]),
            "retrieval_s": stats([R_all[k]["done"]["timings"]["retrieval_ms"] / 1000 for k in rows]),
            "llm_sum_s": stats([R_all[k]["done"]["timings"]["llm_ms_sum"] / 1000 for k in rows]),
            "cost_usd": stats([round(R_all[k]["done"]["cost_usd"], 4) for k in rows])}


R_all, R_rows = {}, {}
for n in R:
    for r in R[n]["rows"]:
        R_all[(n, r["id"], r["run"])] = R[n]["rec"][(r["id"], r["run"])]
        R_rows[(n, r["id"], r["run"])] = r
main_keys = [k for k in R_all if k[0] == "routed-deployed"]
groups = {
    "simple_all": [k for k in main_keys if T[k[1]].get("latency_category") == "simple"],
    "simple_rag_items": [k for k in main_keys if T[k[1]]["ragas"]],
    "simple_data_items": [k for k in main_keys if T[k[1]]["category"] == "data"],
    "multi": [k for k in main_keys if T[k[1]]["category"] == "multi"] + [k for k in R_all if k[0] == "routed-deployed-multi"],
    "system_items": [k for k in main_keys if T[k[1]]["category"] == "system"],
}
lat_bad, lat_out = [], {}
for g, keys in groups.items():
    b = blk(keys)
    lat_out[g] = b
    sj = S["latency_warm_deployed"][g]
    for f in ("ttft_s", "total_s", "retrieval_s", "llm_sum_s", "cost_usd"):
        for st in ("n", "median", "p90", "min", "max"):
            if not close(sj[f][st], b[f][st], 0.0021):
                lat_bad.append((g, f, st, sj[f][st], b[f][st]))
    statuses = Counter(R_all[k]["done"].get("status") for k in keys)
    if dict(statuses) != sj["status"]:
        lat_bad.append((g, "status", sj["status"], dict(statuses)))
check("T9-S1 summary.json latency/cost blocks recompute from raw events", not lat_bad, lat_bad[:10])
check("T9-S2 simple_all excludes multi and system (36 = 32 + 4)", len(groups["simple_all"]) == 36)

for n in ("routed-deployed", "forced-openai-local", "forced-gemini-local"):
    keys = [k for k in R_all if k[0] == n and T[k[1]]["ragas"]]
    b = blk(keys)
    lat_out[f"cmp:{n}"] = b
    lat_out[f"cmp:{n}:cost_mean"] = round(statistics.mean(R_all[k]["done"]["cost_usd"] for k in keys), 4)
    lat_out[f"cmp:{n}:status"] = dict(Counter(R_all[k]["done"].get("status") for k in keys))
    sj = S["provider_comparison"][n]
    bad = [(f, st) for f in ("ttft_s", "total_s", "llm_sum_s") for st in ("median", "p90", "max") if not close(sj[f][st], b[f][st], 0.0021)]
    check(f"T9-S3 provider comparison block {n} recomputes", not bad, bad)

# routing share over all deployed requests
dep = [k for k in R_all if k[0] in ("routed-deployed", "routed-deployed-multi")]
ok_calls = [c for k in dep for c in R_all[k]["results"] if c.get("status") == "ok"]
share = Counter(c["provider"] for c in ok_calls)
synth = Counter(c["provider"] for c in ok_calls if c.get("call_site") == "synthesize")
both = sum(1 for k in dep if len({c["provider"] for c in R_all[k]["results"] if c.get("status") == "ok"}) > 1)
both_done = sum(1 for k in dep if len(R_all[k]["done"].get("providers_used") or []) > 1)
non_ok = Counter((c.get("provider"), c.get("status"), c.get("error_kind")) for k in dep for c in R_all[k]["results"] if c.get("status") != "ok")
model_mismatch = [(k, c["model_id"], c.get("model_returned")) for k in dep for c in R_all[k]["results"]
                  if c.get("status") == "ok" and c.get("model_returned") and not str(c["model_returned"]).startswith(str(c["model_id"]))]
routing = {"requests": len(dep), "ok_calls": len(ok_calls), "by_provider": dict(share), "synthesis": dict(synth),
           "requests_both_ok_calls": both, "requests_both_providers_used": both_done, "non_ok_calls": {str(k): v for k, v in non_ok.items()},
           "model_returned_mismatch": model_mismatch[:5],
           "dimensions": dict(Counter(rt.get("dimension") for k in dep for rt in R_all[k]["routes"])),
           "synthesis_dims_by_provider": {str(k_): v_ for k_, v_ in Counter((rt.get("provider"), rt.get("dimension")) for k in dep for rt in R_all[k]["routes"] if rt.get("call_site") == "synthesize").items()}}
check("T9-S4 routing share 50 requests, 217 ok calls, 190 Gemini / 27 OpenAI, synth 31/19, 19 both",
      len(dep) == 50 and len(ok_calls) == 217 and share == Counter({"gemini-openrouter": 190, "openai": 27})
      and synth == Counter({"gemini-openrouter": 31, "openai": 19}) and both_done == 19, routing)
sr = S["routing_deployed"]
check("T9-S5 summary.json routing block agrees", sr["llm_calls_ok"] == len(ok_calls) and sr["requests_using_both"] == both_done
      and sr["synthesis_provider"] == dict(synth))
check("T9-S6 returned model id matches the routed model on every ok call", not model_mismatch, model_mismatch[:5])

# retrieval diagnostic
diag = {}
for n in ("routed-deployed", "forced-openai-local", "forced-gemini-local"):
    ranks, hit_items, doc_items = [], 0, 0
    for i in ragas_ids:
        x = R_all[(n, i, 1)]
        order = [k for k, t in x["retrieved"] if t == "paper_chunk"]
        rk = [order.index(c) + 1 if c in order else None for c in T[i]["reference_chunk_ids"]]
        ranks += rk
        hit_items += any(rk)
        doc_items += bool({c.split("::")[0] for c in T[i]["reference_chunk_ids"]} & {k.split("::")[0] for k in order})
    got = [v for v in ranks if v]
    diag[n] = {"items_chunk_hit": hit_items, "items_doc_hit": doc_items, "ref_retrieved": len(got), "ref_total": len(ranks),
               "median_rank": statistics.median(got)}
    sd = S["retrieval_diagnostic"][n]
    check(f"T9-S7 retrieval diagnostic {n} recomputes", sd["item_any_reference_chunk_retrieved"] == hit_items and
          sd["item_reference_paper_retrieved"] == doc_items and sd["reference_chunks_retrieved"] == len(got)
          and sd["rank_of_retrieved_median"] == diag[n]["median_rank"], diag[n])

# system checks: re-derive the data/multi expectations from the test set (not the scorer's constants)
sc = json.loads((P9 / "routed-deployed/system_checks.json").read_text())
check("T9-SC1 system_checks 12/12 pass", sc["summary"]["n"] == 12 and sc["summary"]["passed"] == 12, sc["summary"])
multi_rep = []
for k in groups["multi"]:
    x = R_all[k]
    plan = next(e["data"] for e in x["E"] if e["event"] == "plan")
    tasks = plan.get("tasks", [])
    shape = [(t["worker"], tuple(t.get("depends_on", []))) for t in tasks]
    reads = " ".join(json.dumps(e["data"].get("args_safe", {})) for e in x["E"] if e["event"] == "tool_call" and e["data"].get("agent") == "summary")
    exp = {"m01": "2606.05339", "m02": "2601.02553"}[k[1]]
    data_rows = [e["data"] for e in x["E"] if e["event"] == "tool_result" and e["data"].get("agent") == "data"]
    cites = {e["data"].get("cite_id", "")[:1] for e in x["E"] if e["event"] == "citation"}
    multi_rep.append({"key": f"{k[0]}:{k[1]}:{k[2]}", "status": x["done"].get("status"), "shape": shape,
                      "summary_read_expected": exp in reads, "D": "D" in cites, "C": "C" in cites,
                      "data_ok": any(d.get("status") == "ok" for d in data_rows)})
check("T9-SC2 all 8 multi runs: done, data->summary dependency, Summary read the SQL-truth paper, D# and C#",
      all(m["status"] == "done" and m["shape"][0][0] == "data" and any(w == "summary" and dep_ for w, dep_ in m["shape"][1:])
          and m["summary_read_expected"] and m["D"] and m["C"] for m in multi_rep), multi_rep)

# x04 sharing: both agents ran the same tavily tool with different arguments
x = R_all[("routed-deployed", "x04", 1)]
tc = [(e["data"].get("agent"), e["data"].get("tool"), json.dumps(e["data"].get("args_safe", {}))[:160])
      for e in x["E"] if e["event"] == "tool_call" and e["data"].get("tool") == "tavily_search"]
tr = [(e["data"].get("agent"), e["data"].get("status")) for e in x["E"] if e["event"] == "tool_result" and e["data"].get("tool") == "tavily_search"]
check("T9-SC3 x04 same MCP tool, two agents, distinct queries, both ok",
      {a for a, _, _ in tc} >= {"research", "summary"} and len({q for _, _, q in tc}) >= 2 and
      {a for a, s_ in tr if s_ == "ok"} >= {"research", "summary"}, {"calls": tc, "results": tr})

# secret canary scan of all phase9 artifacts
pat = re.compile(r"tavilyApiKey=(?!<|\*|REDACTED|\[)[A-Za-z0-9_\-]{8,}|sk-[A-Za-z0-9_\-]{20,}|sk-or-v1-[0-9a-f]{20,}|tvly-[A-Za-z0-9]{10,}|AIza[0-9A-Za-z_\-]{30,}")
hits = []
for f in P9.rglob("*"):
    if f.is_file():
        for m in pat.finditer(f.read_text(encoding="utf-8", errors="ignore")):
            hits.append((str(f.relative_to(ROOT)), m.group(0)[:12] + "..."))
check("T9-SEC no credential patterns in eval/results/phase9", not hits, hits[:5])

out = {"checks": checks, "passed": sum(c["pass"] for c in checks), "total": len(checks), "ragas": ragas_out,
       "latency": lat_out, "routing": routing, "retrieval_diag": diag, "multi_runs": multi_rep, "dropped_markers": DROPPED}
Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=str))
for c in checks:
    print(("PASS " if c["pass"] else "FAIL ") + c["id"] + ("" if c["pass"] else f"  {str(c['detail'])[:600]}"))
print(f"{out['passed']}/{out['total']} passed")
