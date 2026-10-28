"""Tester Phase 8 retest (P8-D3): Routing-tab geometry with REAL route rows on the deployed app.

ONE ordinary real query (REAL providers; ~$0.03-0.05) after a free 2,500-char HTTP 422 turn head, then the Routing
tab is measured in light and dark mode at 1500 px and 1100 px window widths:
  G1  all columns inside the panel, no horizontal overflow (P8-D2 stays fixed)
  G2  "Turn" header on one line, every "#n" cell on one line (P8-D3)
  G3  no header word broken mid-word; no short word (<= 14 chars) in the Agent / Provider cells broken mid-word
  G4  every row matches its llm_route event (agent/call, provider label, model id, rationale), model_returned == model_id
Usage: python tests/e2e/phase8_tester_routing_geom_real.py BASE_URL OUT_DIR CHROME_PROFILE_DIR
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
from phase8_tester_deployed import STATE, open_tab, routing_checks, streaming_metrics, submit, wait_done  # noqa: E402
import phase8_tester_deployed as dep  # noqa: E402
from phase8_tester_live_nospend import GEOM  # noqa: E402

PORT = 9347
Q = "Which papers in the corpus have 'debate' in the title, and what does each one argue in a sentence or two?"

# For each text word in the given cells, count distinct line tops of its glyph rects (>1 = broken mid-word).
SPLITS = r"""(() => {
  const broken = (cell, maxLen) => { const out = []; const w = document.createTreeWalker(cell, NodeFilter.SHOW_TEXT);
    for (let n; (n = w.nextNode());) { const re = /\S+/g; let m;
      while ((m = re.exec(n.data))) { if (m[0].length > maxLen) continue; const r = document.createRange();
        r.setStart(n, m.index); r.setEnd(n, m.index + m[0].length);
        const tops = new Set([...r.getClientRects()].filter(x => x.width > 0).map(x => Math.round(x.top)));
        if (tops.size > 1) out.push(m[0]); } } return out; };
  const rows = [...document.querySelectorAll('#routing-rows tr:not(.turn-head)')];
  const col = (i) => rows.map(r => r.children[i]).filter(Boolean);
  const agentW = col(1).map(td => Math.round(td.getBoundingClientRect().width));
  return {header_broken: [...document.querySelectorAll('#tab-routing thead th')].flatMap(th => broken(th, 99)),
          agent_broken: col(1).flatMap(td => broken(td, 14)), provider_broken_short: col(2).flatMap(td => broken(td, 14)),
          turn_broken: col(0).flatMap(td => broken(td, 99)), agent_col_px: agentW[0] || null,
          turn_head_lines: (() => { const t = document.querySelector('#routing-rows tr.turn-head td'); if (!t) return null;
            const r = document.createRange(); r.selectNodeContents(t);
            return new Set([...r.getClientRects()].filter(x => x.width > 0).map(x => Math.round(x.top))).size; })()}; })()"""


async def main():
    base, out, profile = sys.argv[1].rstrip("/"), pathlib.Path(sys.argv[2]), sys.argv[3]
    out.mkdir(parents=True, exist_ok=True)
    dep.PORT = PORT
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
                             "--no-first-run", "--no-default-browser-check", "--window-size=1500,1100", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rep = {"base": base, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "real": True, "results": [], "geometry": {}}

    def check(name, ok, detail):
        rep["results"].append({"id": name, "pass": bool(ok), "detail": detail})
        print("PASS" if ok else "FAIL", name)

    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/version")
                break
            except Exception:
                time.sleep(0.2)
        A = await open_tab(base + "/")
        await A.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "light"}])
        await A.js("document.getElementById('q').value = 'x'.repeat(2500); document.getElementById('ask').requestSubmit(); true")
        await asyncio.sleep(3)
        await submit(A, Q)
        s, tap, obs = await wait_done(A)
        rc = routing_checks(s, tap)
        res = {e["d"]["decision_id"]: e["d"] for e in tap["events"] if e["ev"] == "llm_result"}
        rep["turn"] = {"badge": s["badge"], "status": s["status"], "links": s["links"], "stream": streaming_metrics(tap, obs),
                       "routing": rc}
        await A.js("document.querySelector('nav button[data-tab=routing]').click(); true")
        for scheme in ("light", "dark"):
            await A.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": scheme}])
            for width in (1500, 1100):
                await A.call("Emulation.setDeviceMetricsOverride", width=width, height=1100, deviceScaleFactor=1, mobile=False)
                await asyncio.sleep(0.6)
                g, sp = await A.js(GEOM), await A.js(SPLITS)
                key = f"{scheme}_{width}"
                rep["geometry"][key] = {"geom": g, "splits": sp}
                await shot(A, out / f"routing_{key}.png")
                check(f"G1[{key}] columns inside panel, no horizontal overflow",
                      g["route_rows"] > 0 and all(c["right"] <= g["panel_right"] + 1 for c in g["cols"]) and g["scrollW"] <= g["clientW"] + 1,
                      {k: g[k] for k in ("panel_right", "cols", "scrollW", "clientW", "route_rows")})
                check(f"G2[{key}] 'Turn' header and every '#n' cell on one line",
                      g["turn_header_lines"] == 1 and all(c["lines"] == 1 for c in g["first_col_cells"]) and not sp["turn_broken"],
                      {"turn_header_lines": g["turn_header_lines"], "first_col_cells": g["first_col_cells"]})
                check(f"G3[{key}] no header / Agent / short Provider word broken mid-word",
                      not sp["header_broken"] and not sp["agent_broken"] and not sp["provider_broken_short"], sp)
        await A.call("Emulation.clearDeviceMetricsOverride")
        check("G4 routing rows match llm_route events; model_returned == model_id; turn done",
              rc["all_rows_match"] and rc["chips_match"] and all(res.get(p_id, {}).get("model_returned") for p_id in res)
              and all(p["model_returned"] == p["model"] for p in rc["pairs"] if p["model_returned"]) and "done" in s["badge"].lower(),
              {"badge": s["badge"], "providers": rc["providers"], "n_routes": rc["n_routes"], "pairs": rc["pairs"]})
        usage = [e["d"] for e in tap["events"] if e["ev"] == "done"]
        rep["done_event"] = usage[0] if usage else None
        rep["console"] = [json.dumps(e)[:300] for e in A.events if e.get("method") == "Runtime.exceptionThrown"]
        check("G5 no uncaught exceptions", not rep["console"], rep["console"])
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except Exception:
            proc.kill()
        (out / "routing_geom_real_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({"passed": sum(r["pass"] for r in rep["results"]), "failed": sum(not r["pass"] for r in rep["results"])}))


if __name__ == "__main__":
    asyncio.run(main())
