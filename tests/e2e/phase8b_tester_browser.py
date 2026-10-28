"""Tester Phase 8b: REAL deployed page in REAL headless Chrome (CDP). Checks the new Phase 8b trace rows render
from the real events: plan task row with "first calls: ...", tool rows "seeded by the planner / speculative
search, no LLM round", the "speculative" orchestration row (kept/discarded), the worker result row
"evidence only (the synthesizer writes from it)", plus incremental answer growth, clickable citations that open
the viewer, the routing chips, and no console errors.

Usage: python tests/e2e/phase8b_tester_browser.py BASE_URL OUT_DIR CHROME_PROFILE_DIR QUESTION [QUESTION ...]
"""
import asyncio
import json
import pathlib
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from phase2_browser_cdp import CHROME, shot  # noqa: E402
import phase8_tester_deployed as P8  # noqa: E402

P8.PORT = 9351


async def main():
    base, out, profile = sys.argv[1].rstrip("/"), pathlib.Path(sys.argv[2]), sys.argv[3]
    questions = sys.argv[4:]
    out.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={P8.PORT}", f"--user-data-dir={profile}",
                             "--no-first-run", "--no-default-browser-check", "--window-size=1500,1100", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rep = {"base": base, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "turns": []}
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{P8.PORT}/json/version")
                break
            except Exception:
                time.sleep(0.2)
        cdp = await P8.open_tab(base + "/")
        for i, q in enumerate(questions):
            await P8.submit(cdp, q)
            st, tap, obs = await P8.wait_done(cdp)
            await shot(cdp, out / f"b{i + 1}_answer.png")
            evs = tap["events"]
            plan = next((e["d"] for e in evs if e["ev"] == "plan"), {})
            spec = [e["d"] for e in evs if e["ev"] == "speculative_retrieval"]
            seeded = [e["d"] for e in evs if e["ev"] == "tool_call" and e["d"].get("seeded_by")]
            eo = [e["d"] for e in evs if e["ev"] == "worker_result" and e["d"].get("evidence_only")]
            rows = st["trace"]["rows"]
            chk = {
                "plan_first_calls_row": (not any(t.get("seeds") for t in plan.get("tasks", [])))
                or any("first calls:" in r for r in rows),
                "seeded_rows": sum(1 for r in rows if "no LLM round" in r) == len(seeded),
                "speculative_row": (not spec) or any(f"during planning {spec[0]['status']}" in r for r in rows),
                "evidence_only_row": (not eo) or any("evidence only (the synthesizer writes from it)" in r for r in rows),
                "incremental": P8.streaming_metrics(tap, obs),
                "n_links": len(st["links"]),
                "chips": [c["text"] for c in st["chips"]],
                "status": st["status"], "badge": st["badge"],
            }
            # open the first citation in the viewer and compare with the event
            if st["links"]:
                await cdp.js("[...document.querySelectorAll('#turns .turn')].pop().querySelector('.answer a.cite').click(); true")
                await asyncio.sleep(1.5)
                v = await cdp.js(P8.VIEWER)
                first = st["links"][0].strip("[]").split(",")[0].strip()
                ce = next((e["d"] for e in evs if e["ev"] == "citation" and e["d"]["cite_id"] == first), None)
                chk["viewer"] = {"kind": v["kind"], "title_ok": bool(ce) and (ce.get("title") or "")[:30] in v["title"],
                                 "text_len": len(v["text"])}
                await cdp.js("document.querySelector('nav button[data-tab]')?.click(); true")
            chk["console_errors"] = [m for m in cdp.events if m.get("method") == "Runtime.exceptionThrown"
                                     or (m.get("params", {}).get("type") == "error")]
            rep["turns"].append({"question": q, "rows_sample": [r for r in rows if "seeded" in r or "speculative" in r
                                                                or "evidence only" in r or "first calls" in r],
                                 **chk})
            print(json.dumps({k: v for k, v in rep["turns"][-1].items() if k not in ("incremental",)}, default=str)[:1500])
            print(json.dumps(chk["incremental"]))
    finally:
        proc.terminate()
    (out / "browser_report.json").write_text(json.dumps(rep, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
