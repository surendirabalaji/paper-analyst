"""Tester Phase 6 real-browser check (installed Chrome, headless, CDP) against a running app. REAL calls.

Q1 (hand-off): a Data -> Summary/Research question. Checks: streamed growth, D# links in the answer, a
   "handoff" trace row and "data ⇄ SQLite" rows, clicking an sql_row D# shows its SQL + the paper's row +
   a VISIBLE abstract link to arxiv.org/abs/<id>.
Q2 (aggregate + untrusted HTML in the SQL literal): clicking an sql_result D# shows SQL and result rows and
   NO visible abstract link; the HTML payload appears only as text (no injected element, no dialog).
Usage: python tests/e2e/phase6_tester_browser.py BASE_URL OUT_DIR PROFILE_DIR
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

PORT = 9341
Q1 = "Pull up the most recently posted cs.MA paper from 2025 in the collection and summarise its main contribution."
Q2 = ('How many indexed papers have a title containing the exact text <img src=x onerror=alert(1)> '
      'and how many papers does the corpus hold in total?')
VIEW = ("(() => { const l = document.getElementById('viewer-link'); return {title: document.getElementById('viewer-title').textContent,"
        " meta: document.getElementById('viewer-meta').textContent, text: document.getElementById('viewer-text').textContent,"
        " href: l.getAttribute('href'), linkHidden: l.hidden, linkDisplay: getComputedStyle(l).display,"
        " linkVisible: !!(l.offsetWidth || l.offsetHeight || l.getClientRects().length)}; })()")


async def click_cite(cdp, cid):
    await cdp.js(f"[...document.querySelectorAll('#turns .turn:last-child .answer a.cite')].find(a => a.textContent === {json.dumps(cid)}).click(); true")
    await asyncio.sleep(0.5)
    return await cdp.js(VIEW)


async def main():
    base, out, profile = sys.argv[1].rstrip("/"), pathlib.Path(sys.argv[2]), sys.argv[3]
    out.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
                             "--no-first-run", "--no-default-browser-check", "--window-size=1500,1100", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rep = {"base": base, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        for _ in range(50):
            try:
                page = next(p for p in json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list")) if p["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
            cdp = CDP(ws)
            dialogs = []
            orig = cdp.reader

            async def reader():
                async for msg in ws:
                    m = json.loads(msg)
                    if m.get("method") == "Page.javascriptDialogOpening":
                        dialogs.append(m["params"])
                        await ws.send(json.dumps({"id": 99999, "method": "Page.handleJavaScriptDialog", "params": {"accept": True}}))
                    if "id" in m and m["id"] in cdp.pending:
                        cdp.pending.pop(m["id"]).set_result(m)
                    elif m.get("method") in ("Runtime.consoleAPICalled", "Runtime.exceptionThrown"):
                        cdp.events.append(m)
            rd = asyncio.create_task(reader())
            await cdp.call("Page.enable")
            await cdp.call("Runtime.enable")
            await cdp.call("Page.navigate", url=base + "/")
            for _ in range(600):
                if await cdp.js("document.readyState + '|' + !!document.getElementById('ask')") == "complete|true":
                    break
                await asyncio.sleep(0.2)
            for key, q in (("q1", Q1), ("q2", Q2)):
                samples = await ask(cdp, q, timeout=240)
                r = {"query": q, "final": samples[-1],
                     "distinct_streaming_lengths": len({s["len"] for s in samples if s["streaming"] and s["len"] > 0})}
                r["answer"] = await cdp.js("document.querySelector('#turns .turn:last-child .answer').textContent")
                r["links"] = await cdp.js("[...document.querySelectorAll('#turns .turn:last-child .answer a.cite')].map(a => a.textContent)")
                await cdp.js("document.querySelector('nav button[data-tab=trace]').click(); true")
                await asyncio.sleep(0.3)
                r["trace_rows"] = await cdp.js("[...document.querySelectorAll('#trace li')].map(l => l.textContent)"
                                               ".filter(t => t.includes('SQLite') || t.toLowerCase().includes('handoff'))")
                await shot(cdp, out / f"{key}_trace.png")
                r["viewers"] = {}
                for cid in [c for c in r["links"] if c.startswith("D")][:3]:
                    r["viewers"][cid] = await click_cite(cdp, cid)
                    await shot(cdp, out / f"{key}_{cid}_viewer.png")
                for cid in [c for c in r["links"] if c.startswith("C")][:1]:
                    r["viewers"][cid] = await click_cite(cdp, cid)  # regression: C# after D# resets the link
                r["injected_img"] = await cdp.js("document.querySelectorAll('img[src=\"x\"], img[onerror]').length")
                rep[key] = r
            rep["dialogs"] = dialogs
            rep["console_errors"] = [e for e in cdp.events if e.get("method") == "Runtime.exceptionThrown"
                                     or e.get("params", {}).get("type") == "error"]
            rd.cancel()
    finally:
        proc.terminate()
    (out / "browser_report.json").write_text(json.dumps(rep, indent=1, default=str, ensure_ascii=False))
    for k in ("q1", "q2"):
        r = rep.get(k, {})
        print(k, r.get("final"), r.get("distinct_streaming_lengths"), r.get("links"))
        print("  trace:", [t[:160] for t in r.get("trace_rows", [])])
        for c, v in r.get("viewers", {}).items():
            print("  ", c, v["title"][:80], "| link", v["href"], "hidden", v["linkHidden"], "visible", v["linkVisible"], "|", v["text"][:220].replace("\n", " / "))
        print("  injected_img", r.get("injected_img"))
    print("dialogs", rep.get("dialogs"), "console_errors", len(rep.get("console_errors", [])))


if __name__ == "__main__":
    asyncio.run(main())
