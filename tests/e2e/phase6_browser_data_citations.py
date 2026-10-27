"""Phase 6 browser check: real headless Chrome (CDP) against a running app (local or deployed).

Asks one Data -> Summary question, then checks that D# citations are links, that clicking one opens the
source viewer with the SQL, the row and the arXiv abstract link, that the Trace tab shows the
"data ⇄ SQLite" rows and the handoff row, and that the console has no errors. REAL provider calls.

Usage: python tests/e2e/phase6_browser_data_citations.py BASE_URL OUT_DIR CHROME_PROFILE_DIR "question"
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


async def main():
    base, out, profile, q = sys.argv[1].rstrip("/"), pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    out.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
                             "--no-first-run", "--no-default-browser-check", "--window-size=1500,1100", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    report = {"base": base, "query": q, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
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
            for _ in range(600):
                if await cdp.js("document.readyState + '|' + !!document.getElementById('ask')") == "complete|true":
                    break
                await asyncio.sleep(0.2)
            samples = await ask(cdp, q, timeout=240)
            report["final"] = samples[-1]
            report["streamed_lengths"] = len({s["len"] for s in samples if s["streaming"] and s["len"] > 0})
            report["answer_links"] = await cdp.js(
                "[...document.querySelectorAll('#turns .turn:last-child .answer a.cite')].map(a => a.textContent)")
            report["sql_trace_rows"] = await cdp.js(
                "[...document.querySelectorAll('#trace li')].map(l => l.textContent)"
                ".filter(t => t.includes('SQLite') || t.includes('handoff'))")
            web = [c for c in report["answer_links"] if c.startswith("D")]
            if web:
                await cdp.js(f"[...document.querySelectorAll('#turns .turn:last-child .answer a.cite')]"
                             f".find(a => a.textContent === {json.dumps(web[0])}).click(); true")
                await asyncio.sleep(0.5)
                report["viewer"] = await cdp.js(
                    "({title: document.getElementById('viewer-title').textContent,"
                    " meta: document.getElementById('viewer-meta').textContent,"
                    " text: document.getElementById('viewer-text').textContent.slice(0, 200),"
                    " link: document.getElementById('viewer-link').href,"
                    " linkText: document.getElementById('viewer-link').textContent})")
                await shot(cdp, out / "browser_data_citation_viewer.png")
            await cdp.js("document.querySelector('nav button[data-tab=trace]').click(); true")
            await asyncio.sleep(0.3)
            await shot(cdp, out / "browser_trace_sql_handoff.png")
            report["console_errors"] = [e for e in cdp.events if e.get("method") == "Runtime.exceptionThrown"
                                        or (e.get("params", {}).get("type") == "error")]
            rd.cancel()
    finally:
        proc.terminate()
    (out / "browser_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: report[k] for k in ("answer_links", "sql_trace_rows", "viewer", "console_errors",
                                             "streamed_lengths") if k in report}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
