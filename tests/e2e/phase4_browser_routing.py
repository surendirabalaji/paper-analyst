"""Tester Phase 4: real headless Chrome (CDP) against the DEPLOYED URL; routing visibility in the UI.

REAL generation through the public service (2 queries, ~$0.05). No credentials are read.
For each turn records: answer growth samples (incremental rendering), the routing line under the
answer, the Routing-tab rows for that turn (agent/call, provider/model, dimension+rationale+inputs,
result cell), and a screenshot. The UI rows are then compared to the SSE events of the same request
captured from the page's own fetch stream (via a fetch wrapper) so the UI shows the real decisions.

Usage: python tests/e2e/phase4_browser_routing.py BASE_URL OUT_DIR CHROME_PROFILE_DIR "q1" "q2"
"""
import asyncio
import json
import pathlib
import subprocess
import sys
import time
import urllib.request

import websockets

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from phase2_browser_cdp import CDP, CHROME, ask, shot  # noqa: E402

PORT = 9336

ROWS = """(() => [...document.querySelectorAll('#routing-rows tr')].map(tr => ({
  cls: tr.className, cells: [...tr.children].map(td => td.innerText)})))()"""
LINES = """(() => [...document.querySelectorAll('.routing-line')].map(e => e.textContent))()"""


async def main():
    base, out, profile, *qs = sys.argv[1].rstrip("/"), pathlib.Path(sys.argv[2]), sys.argv[3], *sys.argv[4:]
    out.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
                             "--no-first-run", "--no-default-browser-check", "--window-size=1600,1200", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    report = {"base": base, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "turns": []}
    try:
        for _ in range(50):
            try:
                pages = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list"))
                page = next(p for p in pages if p["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
            cdp = CDP(ws)
            rd = asyncio.create_task(cdp.reader())
            await cdp.call("Page.enable")
            await cdp.call("Runtime.enable")
            await cdp.call("Page.navigate", url=base + "/")
            await asyncio.sleep(2.0)
            # Tee the page's own SSE stream (does not alter what the app receives).
            await cdp.js("""(() => { const of = window.fetch; window.__sse = [];
              window.fetch = async (...a) => { const r = await of(...a);
                if (String(a[0]).includes('/api/chat')) { const [x, y] = r.body.tee(); const buf = {t: ''};
                  window.__sse.push(buf); (async () => { const rd = y.getReader(); const dec = new TextDecoder();
                  for (;;) { const {done, value} = await rd.read(); if (done) break; buf.t += dec.decode(value, {stream: true}); } })();
                  return new Response(x, {headers: r.headers, status: r.status}); }
                return r; }; return true; })()""")
            for i, q in enumerate(qs, 1):
                samples = await ask(cdp, q)
                await asyncio.sleep(0.5)
                lens = [s["len"] for s in samples]
                growth = sorted(set(lens))
                rows = await cdp.js(ROWS)
                lines = await cdp.js(LINES)
                sse = await cdp.js(f"window.__sse[{i - 1}] ? window.__sse[{i - 1}].t : ''")
                events = []
                for fr in sse.split("\n\n"):
                    ls = [x for x in fr.split("\n") if x and not x.startswith(":")]
                    if len(ls) >= 2 and ls[0].startswith("event: "):
                        events.append((ls[0][7:], json.loads(ls[1][6:])))
                turn_rows = [r for r in rows if r["cells"] and r["cells"][0] == f"#{i}"]
                routes = [d for e, d in events if e in ("llm_route", "fallback")]
                ui_match = len(turn_rows) == len(routes) and all(
                    d["model_id"] in r["cells"][2] and d["rationale"] in r["cells"][3] and d["dimension"] in r["cells"][3]
                    and d["agent"] in r["cells"][1] and "…" not in r["cells"][4]
                    for d, r in zip(routes, turn_rows))
                await cdp.js("document.querySelector('[data-tab=routing]').click(); true")
                await shot(cdp, out / f"turn{i}_routing.png")
                report["turns"].append({
                    "query": q, "n_samples": len(samples), "distinct_answer_lengths_while_streaming": len(growth),
                    "answer_len_final": lens[-1] if lens else 0, "routing_line": lines[i - 1] if len(lines) >= i else None,
                    "ui_rows": turn_rows, "sse_routes": [{k: d[k] for k in ("agent", "call_site", "provider", "model_id",
                                                                             "dimension", "rationale")} for d in routes],
                    "ui_matches_sse": ui_match,
                    "done": next((d for e, d in events if e == "done"), None),
                })
            rd.cancel()
    finally:
        proc.terminate()
    (out / "browser_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for t in report["turns"]:
        print(t["query"][:60], "| growth", t["distinct_answer_lengths_while_streaming"], "| match", t["ui_matches_sse"],
              "|", t["routing_line"], "| providers", (t["done"] or {}).get("providers_used"))


if __name__ == "__main__":
    asyncio.run(main())
