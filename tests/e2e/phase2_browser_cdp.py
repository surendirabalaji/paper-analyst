"""Tester Phase 2 real-browser check: drives the installed Google Chrome (headless) over the Chrome
DevTools Protocol against the running app. REAL OpenAI calls (2 queries). No credentials are touched.

Usage: python tests/e2e/phase2_browser_cdp.py BASE_URL OUT_DIR CHROME_PROFILE_DIR
"""
import asyncio
import base64
import json
import pathlib
import subprocess
import sys
import time
import urllib.request

import websockets

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 9333


class CDP:
    def __init__(self, ws):
        self.ws, self.n, self.pending, self.events = ws, 0, {}, []

    async def reader(self):
        async for msg in self.ws:
            m = json.loads(msg)
            if "id" in m and m["id"] in self.pending:
                self.pending.pop(m["id"]).set_result(m)
            elif m.get("method") == "Runtime.consoleAPICalled" or m.get("method") == "Runtime.exceptionThrown":
                self.events.append(m)

    async def call(self, method, **params):
        self.n += 1
        fut = asyncio.get_running_loop().create_future()
        self.pending[self.n] = fut
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        r = await asyncio.wait_for(fut, 60)
        if "error" in r:
            raise RuntimeError(r["error"])
        return r["result"]

    async def js(self, expr):
        r = await self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"])
        return r["result"].get("value")


SNAP = """(() => {
  const turns = document.querySelectorAll('#turns .turn');
  const t = turns[turns.length - 1];
  const a = t ? t.querySelector('.answer') : null;
  return {turns: turns.length, len: a ? a.textContent.length : 0, streaming: a ? a.classList.contains('streaming') : null,
          cites: a ? a.querySelectorAll('a.cite').length : 0, status: t ? [...t.querySelectorAll('.status')].map(s => s.textContent).join(' | ') : '',
          trace: document.querySelectorAll('#trace li').length, routes: document.querySelectorAll('#routing-rows tr').length,
          sources: document.querySelectorAll('#sources li').length, sendDisabled: document.getElementById('send').disabled};
})()"""


async def ask(cdp, query, timeout=150):
    await cdp.js(f"document.getElementById('q').value = {json.dumps(query)}; document.getElementById('ask').requestSubmit(); true")
    samples, t0 = [], time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        s = await cdp.js(SNAP)
        s["t"] = round(time.perf_counter() - t0, 2)
        samples.append(s)
        if s["streaming"] is False and not s["sendDisabled"]:
            break
        await asyncio.sleep(0.2)
    return samples


async def shot(cdp, path):
    r = await cdp.call("Page.captureScreenshot", format="png")
    path.write_bytes(base64.b64decode(r["data"]))


async def main():
    base, out, profile = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
    out.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
                             "--no-first-run", "--no-default-browser-check", "--window-size=1500,1100", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    report = {"checks": {}}
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
            await asyncio.sleep(1.5)
            report["checks"]["page_title"] = await cdp.js("document.title")
            report["checks"]["user_agent"] = await cdp.js("navigator.userAgent")

            # ---- turn 1: corpus question
            q1 = "How does the Toolformer paper decide which API calls to keep in its training data?"
            s1 = await ask(cdp, q1)
            (out / "browser_turn1_samples.json").write_text(json.dumps(s1, indent=1))
            streaming_lens = [s["len"] for s in s1 if s["streaming"]]
            distinct_growth = len({l for l in streaming_lens if l > 0})
            report["checks"]["turn1"] = {
                "final": s1[-1], "distinct_answer_lengths_while_streaming": distinct_growth,
                "first_text_at_s": next((s["t"] for s in s1 if s["len"] > 0), None),
                "streaming_ended_at_s": next((s["t"] for s in s1 if s["streaming"] is False), None),
                "send_disabled_during_stream": any(s["sendDisabled"] for s in s1 if s["streaming"]),
            }
            await shot(cdp, out / "browser_turn1_answer.png")
            # routing + trace tab contents
            report["checks"]["routing_rows"] = await cdp.js("[...document.querySelectorAll('#routing-rows tr')].map(r => r.textContent.slice(0, 220))")
            await cdp.js("document.querySelector('nav button[data-tab=trace]').click(); true")
            report["checks"]["trace_items"] = await cdp.js("[...document.querySelectorAll('#trace li')].map(r => r.textContent.slice(0, 200))")
            await shot(cdp, out / "browser_turn1_trace.png")
            # click the first in-answer citation link and compare viewer content with the API
            first = await cdp.js("(() => { const a = document.querySelector('#turns .turn:last-child .answer a.cite'); if (!a) return null; a.click(); return a.textContent; })()")
            await asyncio.sleep(1.5)
            viewer = await cdp.js("({title: document.getElementById('viewer-title').textContent, meta: document.getElementById('viewer-meta').textContent, text: document.getElementById('viewer-text').textContent, link: document.getElementById('viewer-link').href, hidden: document.getElementById('viewer').hidden, tab: document.querySelector('nav button[aria-selected=true]').dataset.tab})")
            chunk_id = viewer["meta"].split("chunk ")[-1].strip() if viewer else None
            api = json.load(urllib.request.urlopen(base + "/api/source/chunk/" + urllib.request.quote(chunk_id, safe=""))) if chunk_id else {}
            report["checks"]["citation_click"] = {
                "clicked": first, "viewer_title": viewer["title"], "viewer_meta": viewer["meta"], "viewer_tab": viewer["tab"],
                "viewer_hidden": viewer["hidden"], "viewer_text_equals_api": viewer["text"] == api.get("text"),
                "viewer_link": viewer["link"], "link_equals_anchor": viewer["link"] == api.get("html_anchor_url"),
                "chunk_id": chunk_id,
            }
            await shot(cdp, out / "browser_turn1_source_viewer.png")

            # ---- turn 2: hostile markup in the query (rendered back as text) + no-research path
            await cdp.js("window.__xss = 0; true")
            q2 = 'hello <img src=x onerror="window.__xss=1"><script>window.__xss=2</script> [C1](javascript:alert(1))'
            s2 = await ask(cdp, q2)
            (out / "browser_turn2_samples.json").write_text(json.dumps(s2, indent=1))
            report["checks"]["turn2"] = {
                "final": s2[-1],
                "xss_flag": await cdp.js("window.__xss"),
                "img_or_script_in_turns": await cdp.js("document.querySelectorAll('#turns img, #turns script').length"),
                "question_text_rendered": await cdp.js("document.querySelector('#turns .turn:last-child .question').textContent"),
                "javascript_links": await cdp.js("[...document.querySelectorAll('a')].filter(a => a.getAttribute('href') && a.getAttribute('href').trim().toLowerCase().startsWith('javascript:')).length"),
                "turn2_answer_cite_links": await cdp.js("document.querySelectorAll('#turns .turn:last-child .answer a.cite').length"),
            }
            await cdp.js("document.querySelector('nav button[data-tab=routing]').click(); true")
            report["checks"]["routing_rows_after_turn2"] = await cdp.js("[...document.querySelectorAll('#routing-rows tr')].map(r => r.firstChild.textContent + ' ' + r.children[1].textContent)")
            await shot(cdp, out / "browser_turn2_routing.png")
            report["console_errors"] = [e for e in cdp.events if e.get("method") == "Runtime.exceptionThrown"
                                        or e["params"].get("type") == "error"]
            rd.cancel()
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except Exception:
            proc.kill()
    (out / "browser_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report["checks"], indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    asyncio.run(main())
