"""Tester Phase 8 retest: zero-LLM-spend checks of the REAL deployed UI (REAL headless Chrome, deployed static files).

  H   deployed /, /static/app.js, md.js, styles.css are byte-identical to `git show HEAD:app/static/...`
  L7  P8-D1 on the live page: Enter (real CDP key events) while a turn is in flight does not start a 2nd request.
      /api/chat is held in the BROWSER with CDP Fetch interception (never sent to the server -> no spend), then
      failed with a network error so the turn settles; a fresh Enter afterwards must start a new request again.
  L12 P8-D2 on the live page: a 2,500-char unbroken question (the real server answers HTTP 422 before any LLM
      call -> no spend) must not widen the Routing table past the panel; also records whether short header
      words ("Turn") get broken letter by letter (side effect of overflow-wrap:anywhere on every cell).

Usage: python tests/e2e/phase8_tester_live_nospend.py BASE_URL OUT_DIR CHROME_PROFILE_DIR
"""
import asyncio
import base64
import hashlib
import json
import pathlib
import subprocess
import sys
import time
import urllib.parse
import urllib.request

import websockets

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from phase2_browser_cdp import CHROME  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
PORT = 9345

GEOM = r"""(() => { const p = document.getElementById('tab-routing'); const panel = document.getElementById('panel').getBoundingClientRect();
  const lines = (node) => { if (!node) return null; const r = document.createRange(); r.selectNodeContents(node);
    const ys = new Set([...r.getClientRects()].filter(x => x.width > 0).map(x => Math.round(x.top))); return ys.size; };
  const th = [...document.querySelectorAll('#tab-routing thead th')].map(h => { const r = h.getBoundingClientRect();
    return {t: h.textContent, left: Math.round(r.left), right: Math.round(r.right), width: Math.round(r.width)}; });
  const firstTds = [...document.querySelectorAll('#routing-rows tr:not(.turn-head) td:first-child')].map(td => ({t: td.textContent, lines: lines(td)}));
  return {panel_right: Math.round(panel.right), cols: th, scrollW: p.scrollWidth, clientW: p.clientWidth,
          turn_header_lines: lines(document.querySelector('#tab-routing thead th')), first_col_cells: firstTds.slice(0, 20),
          route_rows: document.querySelectorAll('#routing-rows tr:not(.turn-head)').length,
          turn_heads: document.querySelectorAll('#routing-rows tr.turn-head').length}; })()"""

STATE = r"""(() => { const t = [...document.querySelectorAll('#turns .turn')].pop();
  return {turns: document.querySelectorAll('#turns .turn').length, send_disabled: document.getElementById('send').disabled,
          q_value: document.getElementById('q').value, badge: t ? (t.querySelector('.badge') || {}).textContent : null,
          status: t ? [...t.querySelectorAll('.status')].map(s => s.textContent).join(' | ') : '',
          running: [...document.querySelectorAll('#turns .turn .badge')].filter(b => /running/.test(b.textContent)).length}; })()"""


class Tab:
    def __init__(self, ws):
        self.ws, self.n, self.pending, self.paused, self.console = ws, 0, {}, [], []

    async def reader(self):
        async for msg in self.ws:
            m = json.loads(msg)
            if "id" in m and m["id"] in self.pending:
                self.pending.pop(m["id"]).set_result(m)
            elif m.get("method") == "Fetch.requestPaused":
                self.paused.append(m["params"])
            elif m.get("method") in ("Runtime.exceptionThrown",) or (
                    m.get("method") == "Runtime.consoleAPICalled" and m["params"].get("type") == "error"):
                self.console.append(json.dumps(m)[:300])

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

    async def enter(self):
        await self.js("document.getElementById('q').focus(); true")
        for typ in ("rawKeyDown", "keyUp"):
            await self.call("Input.dispatchKeyEvent", type=typ, key="Enter", code="Enter", windowsVirtualKeyCode=13,
                            nativeVirtualKeyCode=13)

    async def shot(self, path):
        r = await self.call("Page.captureScreenshot", format="png")
        path.write_bytes(base64.b64decode(r["data"]))


def chat_paused(tab):
    return [p for p in tab.paused if "/api/chat" in p["request"]["url"]]


async def main():
    base, out, profile = sys.argv[1].rstrip("/"), pathlib.Path(sys.argv[2]), sys.argv[3]
    out.mkdir(parents=True, exist_ok=True)
    rep = {"base": base, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "llm_spend": 0, "results": []}

    def check(name, ok, detail):
        rep["results"].append({"id": name, "pass": bool(ok), "detail": detail})
        print("PASS" if ok else "FAIL", name)

    # ---- H: deployed static assets == HEAD
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    rep["head"] = head
    hashes = {}
    for path, f in (("/", "index.html"), ("/static/app.js", "app.js"), ("/static/md.js", "md.js"), ("/static/styles.css", "styles.css")):
        live = urllib.request.urlopen(base + path, timeout=30).read()
        local = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:app/static/{f}"], capture_output=True).stdout
        hashes[f] = {"live": hashlib.sha256(live).hexdigest()[:16], "head": hashlib.sha256(local).hexdigest()[:16]}
    check("H deployed static files byte-identical to HEAD", all(v["live"] == v["head"] for v in hashes.values()), hashes)
    live_js = urllib.request.urlopen(base + "/static/app.js", timeout=30).read().decode()
    live_css = urllib.request.urlopen(base + "/static/styles.css", timeout=30).read().decode()
    check("H2 live app.js has the busy guard; live styles.css has the wrap rule and new --ok",
          "if (busy) return;" in live_js and "overflow-wrap: anywhere" in live_css and "--ok: #236a40" in live_css, {})

    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
                             "--no-first-run", "--no-default-browser-check", "--window-size=1500,1100", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/version")
                break
            except Exception:
                time.sleep(0.2)
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/json/new?{urllib.parse.quote(base + '/', safe='')}", method="PUT")
        t = json.load(urllib.request.urlopen(req))
        tab = Tab(await websockets.connect(t["webSocketDebuggerUrl"], max_size=None))
        rd = asyncio.create_task(tab.reader())
        await tab.call("Page.enable")
        await tab.call("Runtime.enable")
        await tab.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "light"}])
        await tab.call("Page.navigate", url=base + "/")
        await asyncio.sleep(2.5)
        # Hold every /api/chat request inside the browser (never forwarded unless continueRequest is called).
        await tab.call("Fetch.enable", patterns=[{"urlPattern": "*/api/chat*", "requestStage": "Request"}])

        # ---- L7: double submit on the live page
        await tab.js("document.getElementById('q').value = 'retest first question (held in browser)'; true")
        await tab.enter()
        await asyncio.sleep(1.0)
        s1 = await tab.js(STATE)
        n1 = len(chat_paused(tab))
        await tab.js("document.getElementById('q').value = 'retest second question while busy'; true")
        await tab.enter()
        await tab.enter()
        await tab.js("document.getElementById('send').click(); true")
        await tab.js("document.getElementById('ask').requestSubmit(); true")
        await asyncio.sleep(1.5)
        s2 = await tab.js(STATE)
        n2 = len(chat_paused(tab))
        await tab.shot(out / "l7_inflight.png")
        for p in chat_paused(tab):
            await tab.call("Fetch.failRequest", requestId=p["requestId"], errorReason="ConnectionClosed")
        await asyncio.sleep(1.5)
        s3 = await tab.js(STATE)
        # busy must be cleared after the failed turn: a fresh Enter starts exactly one new request
        await tab.enter()
        await asyncio.sleep(1.0)
        n4 = len(chat_paused(tab))
        s4 = await tab.js(STATE)
        for p in chat_paused(tab)[n2:]:
            await tab.call("Fetch.failRequest", requestId=p["requestId"], errorReason="ConnectionClosed")
        await asyncio.sleep(1.5)
        s5 = await tab.js(STATE)
        await tab.shot(out / "l7_settled.png")
        d = {"after_first_enter": {"requests": n1, **s1}, "after_3_more_submit_attempts": {"requests": n2, **s2},
             "after_fail": s3, "after_fresh_enter": {"requests": n4, **s4}, "final": s5,
             "server_contacted": False}
        rep["double_submit_live"] = d
        check("L7a live: one request after first Enter; Ask disabled while in flight", n1 == 1 and s1["send_disabled"], d)
        check("L7b live: Enter x2 + click + requestSubmit while in flight start no 2nd request; typed text kept",
              n2 == 1 and s2["q_value"] == "retest second question while busy", d)
        check("L7c live: after the held request fails the turn settles failed, Ask re-enabled",
              not s3["send_disabled"] and s3["running"] == 0 and "fail" in (s3["badge"] or "").lower(), d)
        check("L7d live: busy flag cleared — a fresh Enter afterwards starts exactly one new request",
              n4 == 2 and s5["running"] == 0 and not s5["send_disabled"], d)

        # ---- L12: long unbroken question -> real HTTP 422 (no LLM call), routing table must stay inside the panel
        await tab.call("Fetch.disable")
        await tab.js("document.getElementById('q').value = 'x'.repeat(2500); document.getElementById('ask').requestSubmit(); true")
        await asyncio.sleep(4)
        s6 = await tab.js(STATE)
        await tab.js("document.querySelector('nav button[data-tab=routing]').click(); true")
        await asyncio.sleep(0.4)
        g = await tab.js(GEOM)
        await tab.shot(out / "l12_routing_after_422.png")
        rep["long_token_live"] = {"state": s6, "geometry": g}
        check("L12a live: 2,500-char turn returns visible HTTP 422 failure", "422" in s6["status"], s6)
        check("L12b live: Routing columns stay inside the panel, no horizontal overflow",
              all(c["right"] <= g["panel_right"] + 1 for c in g["cols"]) and g["scrollW"] <= g["clientW"] + 1, g)
        check("L12c live: 'Turn' header not broken across lines", g["turn_header_lines"] == 1, g)
        rep["console"] = tab.console
        check("L11 live: no uncaught exceptions / console errors other than the expected network failures",
              all("ConnectionClosed" in c or "Failed to fetch" in c or "422" in c for c in tab.console), tab.console)
        rd.cancel()
        await tab.ws.close()
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except Exception:
            proc.kill()
        (out / "live_nospend_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({"passed": sum(r["pass"] for r in rep["results"]), "failed": sum(not r["pass"] for r in rep["results"])}))


if __name__ == "__main__":
    asyncio.run(main())
