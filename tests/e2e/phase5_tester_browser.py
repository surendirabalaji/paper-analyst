"""Tester Phase 5 deployed-browser check: real headless Chrome (CDP) against the public app.

One natural question that should involve both Research and Summary with live web. Checks:
  - incremental rendering (distinct streamed answer lengths while streaming),
  - W# and C# answer markers are links; clicking W# opens the viewer with domain, retrieval time, finding
    agent and an http(s) "Open web page" link; clicking C# opens the viewer with the chunk text,
  - Trace tab rows "<agent> ⇄ MCP ... tavily_search on tavily-remote-mcp (<purpose>)" for both agents,
  - Routing tab rows exist for both providers when used, Sources list entries for W# show the domain,
  - no console errors; no credential shapes in the rendered DOM.
REAL providers and REAL Tavily calls through the deployed app. No credentials are read.

Usage: python tests/e2e/phase5_tester_browser.py BASE_URL OUT_DIR CHROME_PROFILE_DIR "question"
"""
import asyncio
import json
import pathlib
import re
import subprocess
import sys
import time
import urllib.request

import websockets

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from phase2_browser_cdp import CDP, CHROME, ask, shot  # noqa: E402

PORT = 9341


class NetCDP(CDP):
    """Also records Network.dataReceived / responseReceived for the /api/chat stream (arrival timing only)."""

    async def reader(self):
        async for msg in self.ws:
            m = json.loads(msg)
            if "id" in m and m["id"] in self.pending:
                self.pending.pop(m["id"]).set_result(m)
            elif m.get("method") in ("Runtime.consoleAPICalled", "Runtime.exceptionThrown"):
                self.events.append(m)
            elif m.get("method") in ("Network.responseReceived", "Network.dataReceived", "Network.loadingFinished"):
                self.net = getattr(self, "net", [])
                self.net.append({"m": m["method"], "id": m["params"].get("requestId"), "t": time.perf_counter(),
                                 "url": m["params"].get("response", {}).get("url"),
                                 "len": m["params"].get("dataLength")})
LAST = "#turns .turn:last-child .answer"


async def click_and_read(cdp, cite):
    await cdp.js(f"[...document.querySelectorAll('{LAST} a.cite')].find(a => a.textContent === {json.dumps(cite)}).click(); true")
    for _ in range(25):
        await asyncio.sleep(0.2)
        t = await cdp.js("document.getElementById('viewer-text').textContent")
        if t and t != "Loading…":
            break
    return await cdp.js(
        "({title: document.getElementById('viewer-title').textContent,"
        " meta: document.getElementById('viewer-meta').textContent,"
        " text: document.getElementById('viewer-text').textContent.slice(0, 300),"
        " link: document.getElementById('viewer-link').href,"
        " linkText: document.getElementById('viewer-link').textContent})")


async def main():
    base, out, profile, q = sys.argv[1].rstrip("/"), pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    out.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
                             "--no-first-run", "--no-default-browser-check", "--window-size=1500,1100", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rep = {"base": base, "query": q, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checks": {}}
    try:
        for _ in range(50):
            try:
                pages = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list"))
                page = next(p for p in pages if p["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
            cdp = NetCDP(ws)
            rd = asyncio.create_task(cdp.reader())
            await cdp.call("Page.enable")
            await cdp.call("Runtime.enable")
            await cdp.call("Network.enable")
            await cdp.call("Page.navigate", url=base + "/")
            for _ in range(600):
                if await cdp.js("document.readyState + '|' + !!document.getElementById('ask')") == "complete|true":
                    break
                await asyncio.sleep(0.2)
            samples = await ask(cdp, q, timeout=300)
            rep["final_sample"] = samples[-1]
            rep["samples"] = [{k: x[k] for k in ("t", "len", "streaming", "cites")} for x in samples]
            net = getattr(cdp, "net", [])
            chat_id = next((n["id"] for n in net if n["m"] == "Network.responseReceived" and (n["url"] or "").endswith("/api/chat")), None)
            arr = [n for n in net if n["id"] == chat_id and n["m"] == "Network.dataReceived"]
            rep["chat_stream_chunks"] = len(arr)
            rep["chat_stream_arrival_span_s"] = round(arr[-1]["t"] - arr[0]["t"], 2) if arr else None
            rep["streamed_lengths"] = len({s["len"] for s in samples if s["streaming"] and s["len"] > 0})
            links = await cdp.js(f"[...document.querySelectorAll('{LAST} a.cite')].map(a => a.textContent)")
            rep["answer_links"] = links
            rep["unlinked_markers"] = await cdp.js(f"[...document.querySelectorAll('{LAST} .cite-dropped')].map(a => a.textContent)")
            trace = await cdp.js("[...document.querySelectorAll('#trace li')].map(l => l.textContent)")
            rep["mcp_trace_rows"] = [t for t in trace if "⇄ MCP" in t]
            rep["gate_rows"] = [t for t in trace if "tavily_search enabled" in t or "tavily_search not enabled" in t]
            rep["routing_rows"] = await cdp.js("[...document.querySelectorAll('#routing-rows tr')].map(r => r.textContent.replace(/\\s+/g,' ').slice(0,200))")
            rep["sources"] = await cdp.js("[...document.querySelectorAll('#sources li')].map(l => l.textContent)")
            w = [c for c in links if c.startswith("W")]
            c = [x for x in links if x.startswith("C")]
            if w:
                rep["w_viewer"] = await click_and_read(cdp, w[0])
                await shot(cdp, out / "w_viewer.png")
            if c:
                rep["c_viewer"] = await click_and_read(cdp, c[0])
            await cdp.js("document.querySelector('nav button[data-tab=trace]').click(); true")
            await asyncio.sleep(0.3)
            await shot(cdp, out / "trace_mcp.png")
            dom = await cdp.js("document.body.innerText")
            rep["dom_secret_shapes"] = len(re.findall(r"tavilyApiKey=(?!REDACTED)|tvly-(?!REDACTED)[A-Za-z0-9_-]{6,}|sk-[A-Za-z0-9_-]{20,}", dom))
            rep["console_errors"] = [e for e in cdp.events if e.get("method") == "Runtime.exceptionThrown"
                                     or (e.get("params", {}).get("type") == "error")]
            rd.cancel()
    finally:
        proc.terminate()
    ck = rep["checks"]
    agents = {m.split(" ⇄")[0].strip().split()[-1] for m in rep.get("mcp_trace_rows", [])}
    ck["incremental_render"] = rep.get("streamed_lengths", 0) >= 5
    ck["both_agents_mcp_rows"] = {"research", "summary"} <= agents
    ck["mcp_rows_name_server_and_tool"] = all("tavily_search on tavily-remote-mcp" in m for m in rep.get("mcp_trace_rows", [])) and bool(rep.get("mcp_trace_rows"))
    wv = rep.get("w_viewer") or {}
    ck["w_link_opens_viewer"] = bool(wv) and wv["link"].startswith("http") and "retrieved" in wv["meta"] \
        and "via tavily_search" in wv["meta"] and wv["linkText"] == "Open web page" and len(wv["text"]) > 20
    cv = rep.get("c_viewer") or {}
    ck["c_link_opens_chunk"] = bool(cv) and "chunk" in cv["meta"] and len(cv["text"]) > 50 and "arxiv.org" in cv["link"]
    ck["no_console_errors"] = rep.get("console_errors") == []
    ck["no_secret_shapes_in_dom"] = rep.get("dom_secret_shapes") == 0
    (out / "browser_report.json").write_text(json.dumps(rep, indent=1, default=str, ensure_ascii=False))
    print(json.dumps({"checks": ck, "agents_with_mcp_rows": sorted(agents), "links": rep.get("answer_links"),
                      "mcp_rows": rep.get("mcp_trace_rows"), "w_viewer": wv, "c_viewer_meta": cv.get("meta"),
                      "routing_rows": rep.get("routing_rows")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
