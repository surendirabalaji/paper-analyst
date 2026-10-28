#!/usr/bin/env python3
"""Phase 9 deterministic checks for the non-RAGAS test items (no LLM calls).

  data    exact match: every expected value appears in the answer (numbers as whole tokens, names verbatim),
          plus a D# citation and a Data task in the plan.
  multi   hand-off: plan is Data -> Summary with a dependency; Summary read the expected paper; D# and C#
          citations; required-fact keywords present in the answer (keyword rubric below).
  system  trace assertions per kind (see CHECKS).

Each check is pass/fail with the evidence that decided it. Items whose expected behaviour needs judgement
(x05, x06) also get `needs_review: true`; the reviewer's verdict is recorded in docs/evaluation.md.

Usage: .venv/bin/python scripts/check_system.py RUN_DIR [RUN_DIR ...]   -> RUN_DIR/system_checks.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "eval/testset/testset_v1.jsonl").read_text().splitlines() if l.strip()}

DATA_EXPECT = {
    "d01": ["14"],
    "d02": ["Guanzhi Wang", "Yuqi Xie", "Yunfan Jiang", "Ajay Mandlekar", "Chaowei Xiao", "Yuke Zhu", "Linxi Fan", "Anima Anandkumar"],
    "d03": ["cs.CL", "47"],
    "d04": ["58"],
}
MULTI_EXPECT = {  # paper Summary must read, and keyword groups for the required facts (any alternative counts)
    "m01": ("2606.05339", [["2606.05339", "Taxonomy of Runtime Faults"], ["833"], ["473"], ["11 "], ["27 "], ["55"]]),
    "m02": ("2601.02553", [["SimpleMem"], ["2601.02553", "January 5, 2026", "2026-01-05", "Jan"],
                           ["compression"], ["26.4"], ["30-fold", "30x", "30×", "30 times", "thirtyfold"]]),
}


def load(run_dir: Path, item_id: str, run: int = 1) -> list[dict]:
    p = run_dir / f"{item_id}.run{run}.events.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def answer_of(ev): return "".join(e["data"]["delta"] for e in ev if e["event"] == "token")
def of(ev, name): return [e["data"] for e in ev if e["event"] == name]


def has_number(text: str, n: str) -> bool:
    return re.search(r"(?<![\d.])" + re.escape(n) + r"(?![\d])", text) is not None


def check(item: dict, ev: list[dict]) -> dict:
    ans = answer_of(ev)
    cites = of(ev, "citation")
    types = [c.get("type") for c in cites]
    plan = (of(ev, "plan") or [{}])[0]
    workers = [t["worker"] for t in plan.get("tasks", [])]
    tools = of(ev, "tool_result")
    done = (of(ev, "done") or [{}])[-1]
    checks: dict[str, bool] = {}
    iid = item["id"]
    if item["category"] == "data":
        exp = DATA_EXPECT[iid]
        for v in exp:
            checks[f"answer has {v!r}"] = has_number(ans, v) if v.isdigit() else v in ans
        checks["plan has a data task"] = "data" in workers
        checks["D# citation"] = any(t and t.startswith("sql") for t in types) or any(c.get("cite_id", "").startswith("D") for c in cites)
    elif item["category"] == "multi":
        paper, groups = MULTI_EXPECT[iid]
        tasks = plan.get("tasks", [])
        checks["plan is data then summary with a dependency"] = (
            len(tasks) >= 2 and tasks[0]["worker"] == "data"
            and any(t["worker"] == "summary" and tasks[0]["id"] in t.get("depends_on", []) for t in tasks[1:]))
        reads = [e["data"] for e in ev if e["event"] == "tool_call" and e["data"].get("agent") == "summary"]
        checks[f"summary read {paper}"] = any(paper in json.dumps(r.get("args_safe", {})) for r in reads)
        checks["D# citation"] = any(c.get("cite_id", "").startswith("D") for c in cites)
        checks["C# citation"] = any(c.get("cite_id", "").startswith("C") for c in cites)
        for g in groups:
            checks[f"fact {g[0]!r}"] = any((has_number(ans, x) if x.replace(".", "").isdigit() else x.lower() in ans.lower()) for x in g)
    else:
        kind = item["kind"]
        c_cites = [c for c in cites if c.get("type") == "paper_chunk"]
        w_cites = [c for c in cites if c.get("type") == "web"]
        if kind == "out_of_corpus":
            checks["no C# corpus citation"] = not c_cites
            checks["web claims carry W# with a retrieval date"] = (not w_cites) or all(c.get("retrieved_at") for c in w_cites)
            checks["answer marks the source as web, not the corpus"] = bool(re.search(r"web|corpus|indexed", ans, re.I))
        elif kind == "missing_paper":
            checks["says no such paper"] = bool(re.search(r"no paper|not in the|does not (exist|answer)|no .*Voyager-2", ans, re.I))
            checks["no citation to a 'Voyager-2' source"] = not any("voyager-2" in (c.get("title", "") or "").lower() for c in cites)
        elif kind == "live":
            live = [t for t in tools if t.get("tool") == "tavily_search" and t.get("status") == "ok"]
            checks["tavily_search ok"] = bool(live)
            checks["W# citations with retrieval date"] = bool(w_cites) and all(c.get("retrieved_at") for c in w_cites)
        elif kind == "mcp_sharing":
            agents = {t.get("agent") for t in tools if t.get("tool") == "tavily_search" and t.get("status") == "ok"}
            checks["research called tavily_search"] = "research" in agents
            checks["summary called tavily_search"] = "summary" in agents
            checks["W# and C# citations"] = bool(w_cites) and bool(c_cites)
        elif kind == "insufficient_evidence":
            checks["states the evidence does not answer it"] = bool(re.search(r"does not answer|no .*(comparable|ranking)|lacks", ans, re.I))
        elif kind == "adversarial":
            checks["refuses the unsupported claim"] = bool(re.search(r"does not answer|no citation|no evidence|not supported|contains no", ans, re.I))
        checks["no unresolved citation markers"] = not done.get("unresolved_citations")
    return {"id": iid, "category": item["category"], "kind": item.get("kind"), "status": done.get("status"),
            "pass": all(checks.values()) if checks else None, "checks": checks,
            "needs_review": item.get("kind") in ("insufficient_evidence", "adversarial", "out_of_corpus"),
            "answer_excerpt": ans[:400]}


def main(dirs: list[str]) -> int:
    for d in dirs:
        run_dir = Path(d)
        res = json.loads((run_dir / "results.json").read_text())["results"]
        rows = []
        for r in sorted(res, key=lambda x: (x["id"], x["run"])):
            item = TESTS[r["id"]]
            if item["ragas"]:
                continue
            row = check(item, load(run_dir, r["id"], r["run"]))
            row["run"] = r["run"]
            rows.append(row)
            print(f"{row['id']} run{row['run']}: {'PASS' if row['pass'] else 'FAIL'} {[k for k, v in row['checks'].items() if not v]}")
        summary = {"n": len(rows), "passed": sum(1 for x in rows if x["pass"]),
                   "by_category": {c: f"{sum(1 for x in rows if x['category'] == c and x['pass'])}/{sum(1 for x in rows if x['category'] == c)}"
                                   for c in sorted({x["category"] for x in rows})}}
        (run_dir / "system_checks.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False))
        print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
