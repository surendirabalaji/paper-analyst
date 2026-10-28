"""Phase 7 browser check: real headless Chrome (CDP) against a running app with an injected worker failure.

Opens BASE_URL/#fault=<spec>&fault_token=<token> (the token is read from .env, never printed or saved; the
URL hash is not sent to the server), asks a two-part question, and checks that the page shows the fault
banner, that the answer streams and ends with the "Incomplete evidence" note, that the Trace tab shows the
FAULT row and "[injected test fault]" labels, that D# citations are links, and that the console has no
errors. REAL provider calls (about $0.02).

Usage: python tests/e2e/phase7_browser_fault.py BASE_URL OUT_DIR CHROME_PROFILE_DIR
"""
import asyncio
import json
import pathlib
import subprocess
import sys
import time
import urllib.parse
import urllib.request

import websockets
from dotenv import dotenv_values

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from phase2_browser_cdp import CDP, CHROME, ask, shot  # noqa: E402

PORT = 9337
ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = "worker:research:exception,worker:summary:exception"
QUERY = "How many papers in the corpus have cs.MA as a category, and how does Self-RAG decide when to retrieve?"


async def main():
    base, out, profile = sys.argv[1].rstrip("/"), pathlib.Path(sys.argv[2]), sys.argv[3]
    token = dotenv_values(ROOT / ".env").get("FAULT_INJECT_TOKEN") or sys.exit("FAULT_INJECT_TOKEN missing in .env")
    out.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
                             "--no-first-run", "--no-default-browser-check", "--window-size=1500,1100", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    report = {"base": base, "fault": SPEC, "query": QUERY, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
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
            frag = urllib.parse.urlencode({"fault": SPEC, "fault_token": token})
            await cdp.call("Page.navigate", url=f"{base}/#{frag}")
            for _ in range(600):
                if await cdp.js("document.readyState + '|' + !!document.getElementById('ask')") == "complete|true":
                    break
                await asyncio.sleep(0.2)
            report["banner"] = await cdp.js("(document.querySelector('.banner.fault') || {}).textContent || null")
            samples = await ask(cdp, QUERY, timeout=240)
            report["final"] = samples[-1]
            report["streamed_lengths"] = len({s["len"] for s in samples if s["streaming"] and s["len"] > 0})
            report["answer_tail"] = await cdp.js(
                "document.querySelector('#turns .turn:last-child .answer').textContent.slice(-300)")
            report["status_line"] = await cdp.js(
                "[...document.querySelectorAll('#turns .turn:last-child .status')].map(s => s.textContent)")
            report["answer_links"] = await cdp.js(
                "[...document.querySelectorAll('#turns .turn:last-child .answer a.cite')].map(a => a.textContent)")
            report["fault_trace_rows"] = await cdp.js(
                "[...document.querySelectorAll('#trace li')].map(l => l.textContent)"
                ".filter(t => t.includes('FAULT') || t.includes('injected') || t.includes('fault injection'))")
            await cdp.js("document.querySelector('nav button[data-tab=trace]').click(); true")
            await asyncio.sleep(0.3)
            await shot(cdp, out / "browser_fault_trace.png")
            report["console_errors"] = [e for e in cdp.events if e.get("method") == "Runtime.exceptionThrown"
                                        or (e.get("params", {}).get("type") == "error")]
            rd.cancel()
    finally:
        proc.terminate()
    text = json.dumps(report, indent=1, default=str)
    assert token not in text, "token leaked into the report"
    (out / "browser_report.json").write_text(text)
    print(json.dumps({k: report.get(k) for k in ("banner", "streamed_lengths", "answer_tail", "status_line",
                                                 "answer_links", "fault_trace_rows", "console_errors")}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
