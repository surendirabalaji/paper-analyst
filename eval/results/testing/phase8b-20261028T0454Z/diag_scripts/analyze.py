import asyncio, json, pathlib, sys
sys.path.insert(0, "tests/e2e")
import phase8b_tester_deployed as D
import httpx
base, out = sys.argv[1], pathlib.Path(sys.argv[2])
cats = {k: c for c, lst in (("simple", D.SIMPLE), ("multi", D.MULTI), ("edge", D.EDGE), ("concurrent", D.CONCURRENT)) for k, _ in lst}
cats.update({k: "fault" for k, _, _ in D.FAULTS})
qs = {k: q for lst in (D.SIMPLE, D.MULTI, D.EDGE, D.CONCURRENT) for k, q in lst}; qs.update({k: q for k, q, _ in D.FAULTS})
async def main():
    rows = []
    async with httpx.AsyncClient(timeout=60) as c:
        for f in sorted(out.glob("*.run1.events.jsonl")):
            key = f.name.split(".run1")[0]
            if key == "warmup": continue
            ev = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            tok = next((e["client_s"] for e in ev if e["event"] == "token"), None)
            done = next((e["client_s"] for e in ev if e["event"] == "done"), None)
            res = {"events": ev, "ttft_s": tok, "total_s": done, "raw": f.name}
            g = await D.generic_checks(c, base, res)
            row = {"key": key, "category": cats.get(key), "query": qs.get(key), "raw": f.name,
                   **{k: v for k, v in g.items() if k != "answer"}, "case": D.case_checks(key, res, g), "answer": g["answer"]}
            rows.append(row)
    (out / "analysis.json").write_text(json.dumps({"runs": rows, "results": [{"raw": r["raw"], "query": r["query"]} for r in rows]}, indent=2, default=str))
    for r in rows:
        print(json.dumps({k: r[k] for k in ("key","status","workers","ttft_s","spec","spec_consistent","seeds_match_plan","evidence_only_consistent","route_fields_ok","model_returned_matches","synth_provider","n_token_events","n_citations","marked_without_citation_event","citation_resolution_failures","unresolved_citations","cost_usd")}))
        if r["case"]: print("   case:", json.dumps(r["case"], default=str)[:900])
asyncio.run(main())
