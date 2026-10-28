import json, sys, statistics, pathlib
d = pathlib.Path(sys.argv[1])
rows = []
for f in sorted(d.glob("*.events.jsonl")):
    if f.name.startswith("warmup"): continue
    ev = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    tok = next((e for e in ev if e["event"] == "token"), None)
    done = next((e for e in ev if e["event"] == "done"), None)
    plan = next((e for e in ev if e["event"] == "plan"), None)
    meta = next((e for e in ev if e["event"] == "meta"), None)
    workers = [t["worker"] for t in plan["data"]["tasks"]] if plan else None
    synth = [e["data"]["provider"] for e in ev if e["event"] == "llm_route" and e["data"]["call_site"].startswith("synth")]
    rows.append((f.name, tok and tok["client_s"], done and done["client_s"], workers, synth,
                 done and done["data"].get("status"), done and done["data"].get("cost_usd"),
                 (meta or {}).get("data", {}).get("revision") if meta else None))
for r in rows: print(r)
simple = [r[1] for r in rows if not r[0].startswith(("cs_ma", "selfrag_vs")) and r[1]]
print("simple n", len(simple), "median TTFT", statistics.median(simple), "min", min(simple), "max", max(simple))
