"""Tester Phase 8, REAL services, two browser checks the deployed run could not force:

1. drop:  real headless Chrome -> local cutting proxy -> the DEPLOYED Cloud Run app. The proxy forwards the page and
          the real SSE stream, then aborts the client connection after the 3rd `token` frame (a genuine mid-stream
          network loss). The UI must settle on a visible error and keep the partial answer.
2. fault: real headless Chrome -> a LOCAL app process started by this script with FAULT_INJECT=worker:data:exception
          (process-wide, local only; nothing deployed is touched). A question needing Data + Research must render a
          partial turn with injected badges and an honest note.

Usage: python tests/e2e/phase8_tester_drop_and_fault.py DEPLOYED_URL OUT_DIR CHROME_PROFILE_DIR
"""
import asyncio
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import urllib.request

import httpx
import uvicorn
import websockets
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from phase2_browser_cdp import CDP, CHROME, shot  # noqa: E402
from phase8_tester_deployed import STATE, TAP  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROXY_PORT, LOCAL_PORT, CDP_PORT = 8795, 8796, 9345
Q_DROP = "In two or three sentences, what is the core idea behind Voyager's skill library?"
Q_FAULT = "How many corpus papers are tagged cs.MA, and what does Reflexion mean by verbal self-reflection?"
UPSTREAM = sys.argv[1].rstrip("/")
proxy = FastAPI()
PROXY_LOG = {}


@proxy.get("/{path:path}")
async def get_any(path: str):
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.get(f"{UPSTREAM}/{path}")
    return Response(r.content, r.status_code, media_type=r.headers.get("content-type"))


@proxy.post("/api/chat")
async def chat(req: Request):
    body = await req.body()
    client = httpx.AsyncClient(timeout=None)
    up = await client.send(client.build_request("POST", f"{UPSTREAM}/api/chat", content=body,
                                                headers={"content-type": "application/json", "accept": "text/event-stream"}), stream=True)
    PROXY_LOG["request_id"] = up.headers.get("x-request-id")

    async def gen():
        n = 0
        try:
            async for chunk in up.aiter_bytes():
                yield chunk
                n += chunk.count(b"event: token")
                if n >= 3:
                    PROXY_LOG["cut_after_token_frames"] = n
                    PROXY_LOG["cut_at"] = time.strftime("%H:%M:%SZ", time.gmtime())
                    raise ConnectionAbortedError("tester proxy: simulated network loss")
        finally:
            await up.aclose()
            await client.aclose()

    return StreamingResponse(gen(), media_type="text/event-stream")


async def page(cdp_port, url):
    req = urllib.request.Request(f"http://127.0.0.1:{cdp_port}/json/new?about:blank", method="PUT")
    t = json.load(urllib.request.urlopen(req))
    ws = await websockets.connect(t["webSocketDebuggerUrl"], max_size=None)
    cdp = CDP(ws)
    cdp._rd = asyncio.create_task(cdp.reader())
    await cdp.call("Page.enable")
    await cdp.call("Runtime.enable")
    await cdp.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "light"}])
    await cdp.call("Page.navigate", url=url)
    await asyncio.sleep(2)
    await cdp.js(TAP)
    return cdp


async def ask(cdp, q, timeout=170):
    await cdp.js(TAP)
    await cdp.js(f"document.getElementById('q').value = {json.dumps(q)}; document.getElementById('ask').requestSubmit(); true")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        await asyncio.sleep(0.5)
        s = await cdp.js("({s: [...document.querySelectorAll('#turns .turn')].pop().querySelector('.answer').classList.contains('streaming'), d: document.getElementById('send').disabled})")
        if not s["s"] and not s["d"]:
            break
    await asyncio.sleep(0.5)
    return await cdp.js(STATE), await cdp.js("window.__tap")


async def main():
    out, profile = pathlib.Path(sys.argv[2]), sys.argv[3]
    out.mkdir(parents=True, exist_ok=True)
    rep = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    srv = uvicorn.Server(uvicorn.Config(proxy, host="127.0.0.1", port=PROXY_PORT, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    env = {**os.environ, "FAULT_INJECT": "worker:data:exception", "PORT": str(LOCAL_PORT)}
    env.pop("FAULT_INJECT_TOKEN", None)
    log = open(out / "local_server.log", "w")
    app_proc = subprocess.Popen([str(ROOT / ".venv/bin/uvicorn"), "app.main:app", "--host", "127.0.0.1", "--port", str(LOCAL_PORT)],
                                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    chrome = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={CDP_PORT}", f"--user-data-dir={profile}",
                               "--no-first-run", "--no-default-browser-check", "--window-size=1500,1100", "about:blank"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(150):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{LOCAL_PORT}/health", timeout=2)
                urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json/version", timeout=2)
                break
            except Exception:
                time.sleep(0.3)
        # 1. genuine mid-stream connection loss against the deployed app
        A = await page(CDP_PORT, f"http://127.0.0.1:{PROXY_PORT}/")
        s, tap = await ask(A, Q_DROP)
        rep["drop"] = {"proxy": PROXY_LOG, "state": {k: s[k] for k in ("answer", "status", "status_cls", "badge", "send_disabled", "streaming", "links")},
                       "tokens_received": sum(e["ev"] == "token" for e in tap["events"]), "got_done": any(e["ev"] == "done" for e in tap["events"]),
                       "trace_tail": s["trace"]["rows"][-2:], "trace_badge": s["trace"]["summary"][:40]}
        await shot(A, out / "f1_drop_deployed.png")
        # 2. injected Data failure in a local real run
        B = await page(CDP_PORT, f"http://127.0.0.1:{LOCAL_PORT}/")
        s, tap = await ask(B, Q_FAULT)
        rep["fault"] = {"state": {k: s[k] for k in ("answer", "status", "status_cls", "badge", "links", "chips")},
                        "trace": s["trace"], "fault_events": [e["d"] for e in tap["events"] if e["ev"] == "fault_injected"],
                        "worker_results": [e["d"] for e in tap["events"] if e["ev"] == "worker_result"],
                        "done": next((e["d"] for e in tap["events"] if e["ev"] == "done"), None)}
        await shot(B, out / "f2_fault_partial_chat.png")
        await B.js("document.querySelector('nav button[data-tab=trace]').click(); true")
        await shot(B, out / "f2_fault_partial_trace.png")
        rep["console"] = [json.dumps(e)[:300] for c in (A, B) for e in c.events
                          if e.get("method") == "Runtime.exceptionThrown" or e["params"].get("type") == "error"]
    finally:
        chrome.terminate()
        app_proc.terminate()
        try:
            app_proc.wait(10)
        except Exception:
            app_proc.kill()
        srv.should_exit = True
        log.close()
        (out / "drop_fault_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    print("written")


if __name__ == "__main__":
    asyncio.run(main())
