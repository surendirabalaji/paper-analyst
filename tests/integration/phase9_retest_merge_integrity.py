#!/usr/bin/env python3
"""Phase 9 retest (P9-D1): offline integrity check of the merged RAGAS rescore.

Compares the committed ragas.json / summary.json with the pre-fix versions at git revision BASE (default 765584c):
- the set of rescored rows is exactly the set of RAGAS rows whose raw events contain ok `web_search` results;
- rescored rows changed only the context-dependent metrics they list, kept answer_relevancy, and their
  context count now equals paper chunks + web results recomputed independently from the raw events;
- every other row in every config is byte-identical in all metric and context fields;
- summary.json differs only in blocks derived from the rescored metrics.
No network, no LLM calls.

Usage: .venv/bin/python tests/integration/phase9_retest_merge_integrity.py OUT_JSON [BASE]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P9 = "eval/results/phase9"
BASE = sys.argv[2] if len(sys.argv) > 2 else "765584c"
CONFIGS = ("routed-deployed", "forced-openai-local", "forced-gemini-local")
CTX_METRICS = {"faithfulness", "context_precision", "context_recall"}
checks: list[dict] = []


def check(cid, ok, detail=""):
    checks.append({"id": cid, "pass": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + cid + ("" if ok else f"  {detail}"), flush=True)


def old(path):
    return json.loads(subprocess.check_output(["git", "-C", str(ROOT), "show", f"{BASE}:{path}"]))


def events(cfg, raw):
    return [json.loads(l) for l in (ROOT / P9 / cfg / raw).read_text(encoding="utf-8").splitlines() if l.strip()]


def indep_contexts(E):
    seen, n_pc, n_web = set(), 0, 0
    for e in E:
        if e["event"] == "retrieval":
            for c in e["data"].get("chunks", []):
                k = c.get("chunk_id") or c.get("source_url") or c.get("cite_id")
                if k and k not in seen:
                    seen.add(k)
                    n_pc += c.get("type") == "paper_chunk"
        elif e["event"] == "web_search" and e["data"].get("status") == "ok":
            for r in e["data"].get("results", []):
                k = r.get("url") or r.get("cite_id")
                if k and k not in seen:
                    seen.add(k)
                    n_web += 1
    return n_pc, n_web


report = {"base": BASE, "configs": {}}
for cfg in CONFIGS:
    new_j = json.loads((ROOT / P9 / cfg / "ragas.json").read_text())
    old_j = old(f"{P9}/{cfg}/ragas.json")
    meta = json.loads((ROOT / P9 / cfg / "results.json").read_text())
    raw_of = {(r["id"], r["run"]): r["raw"] for r in meta["results"]}
    O = {(r["id"], r["run"]): r for r in old_j["rows"]}
    N = {(r["id"], r["run"]): r for r in new_j["rows"]}
    check(f"RT9-{cfg}-keys same row set before/after", set(O) == set(N), sorted(set(O) ^ set(N)))
    web_rows = set()
    for k in N:
        _, nw = indep_contexts(events(cfg, raw_of[k]))
        if nw:
            web_rows.add(k)
    rescored = {k for k, r in N.items() if r.get("rescored_metrics")}
    check(f"RT9-{cfg}-W rescored rows == rows with ok web_search results", rescored == web_rows,
          {"rescored": sorted(rescored), "web": sorted(web_rows)})
    listed = set(new_j["summary"].get("rescored_after_P9_D1", []))
    check(f"RT9-{cfg}-L summary rescored list matches rows", listed == {f"{i}.run{n}" for i, n in rescored}, sorted(listed))
    unchanged_bad, rescored_bad, detail = [], [], {}
    fields = ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "n_contexts", "n_contexts_all",
              "answer_chars", "status", "category")
    for k, o in O.items():
        n = N.get(k, {})
        if k not in rescored:
            diffs = [f for f in fields if o.get(f) != n.get(f)]
            if diffs:
                unchanged_bad.append((k, diffs))
            continue
        ms = set(n["rescored_metrics"])
        n_pc, n_web = indep_contexts(events(cfg, raw_of[k]))
        d = {f: (o.get(f), n.get(f)) for f in fields if o.get(f) != n.get(f)}
        detail[f"{k[0]}.run{k[1]}"] = {"rescored_metrics": sorted(ms), "changed": d, "indep_paper": n_pc, "indep_web": n_web}
        if not ms <= CTX_METRICS:
            rescored_bad.append((k, "non-context metric rescored", sorted(ms)))
        if o.get("answer_relevancy") != n.get("answer_relevancy"):
            rescored_bad.append((k, "answer_relevancy changed"))
        for m in ("faithfulness", "context_precision", "context_recall"):
            if m not in ms and o.get(m) != n.get(m):
                rescored_bad.append((k, f"{m} changed but not listed"))
            if m in ms and m in o and n.get(m) is None:
                rescored_bad.append((k, f"{m} missing after rescore"))
        if n.get("n_contexts_all") != n_pc + n_web or n.get("n_contexts") != n_pc + n_web:
            rescored_bad.append((k, "context count", n.get("n_contexts_all"), n_pc, n_web))
        if o.get("answer_chars") != n.get("answer_chars"):
            rescored_bad.append((k, "answer changed"))
        if n.get("errors"):
            rescored_bad.append((k, "errors", n["errors"]))
    check(f"RT9-{cfg}-U non-rescored rows unchanged", not unchanged_bad, unchanged_bad[:10])
    check(f"RT9-{cfg}-R rescored rows: only listed context metrics changed, AR kept, contexts = chunks + web",
          not rescored_bad, rescored_bad)
    report["configs"][cfg] = {"rescored": detail, "judge_cost_old": old_j["summary"].get("judge_cost_usd_est"),
                              "judge_cost_new": new_j["summary"].get("judge_cost_usd_est"),
                              "all_old": old_j["summary"]["all"], "all_new": new_j["summary"]["all"]}

# summary.json: which leaves changed
def leaves(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{p}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{p}[{i}]")
    else:
        yield p, x


so = dict(leaves(old(f"{P9}/summary.json")))
sn = dict(leaves(json.loads((ROOT / P9 / "summary.json").read_text())))
changed = sorted(k for k in set(so) | set(sn) if so.get(k) != sn.get(k))
allowed = ("faithfulness", "context_precision", "context_recall", "judge_cost", "generated", "rescored", "contexts",
           "wall_s")  # wall_s: overwritten by the merge call (retest note P9R-N1), reported separately
bad = [k for k in changed if not any(a in k for a in allowed)]
check("RT9-SUM summary.json changed only in context-metric / judge-cost / provenance leaves", not bad, bad[:20])
report["summary_changed_leaves"] = {k: (so.get(k), sn.get(k)) for k in changed}

out = {"checks": checks, "passed": sum(c["pass"] for c in checks), "total": len(checks), **report}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=str))
print(f"{out['passed']}/{out['total']} passed")
sys.exit(0 if out["passed"] == out["total"] else 1)
