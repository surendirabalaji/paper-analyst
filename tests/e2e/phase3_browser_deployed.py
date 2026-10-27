"""Tester Phase 3: real headless Chrome (CDP) against the DEPLOYED Cloud Run URL.

REAL OpenAI generation + embeddings through the public service (2 queries). No credentials are read.
Records page-load timing (cold or warm, as established separately from Cloud Run logs), static asset
responses, streamed answer growth, routing/trace rows, citation click -> source viewer vs. API, and an
off-corpus turn (no invented citations).

Usage: python tests/e2e/phase3_browser_deployed.py BASE_URL OUT_DIR CHROME_PROFILE_DIR "corpus q" "off-corpus q"
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

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from phase2_browser_cdp import CDP, CHROME, SNAP, ask, shot  # noqa: E402

PORT = 9334


async def main():
    base, out, profile, q1, q2 = sys.argv[1].rstrip("/"), pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
    out.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
                             "--no-first-run", "--no-default-browser-check", "--window-size=1500,1100", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    report = {"base": base, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checks": {}}
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
            responses = []
            orig_reader = cdp.reader

            async def reader():  # also keep Network.responseReceived
                async for msg in cdp.ws:
                    m = json.loads(msg)
                    if "id" in m and m["id"] in cdp.pending:
                        cdp.pending.pop(m["id"]).set_result(m)
                    elif m.get("method") == "Network.responseReceived":
                        r = m["params"]["response"]
                        responses.append({"url": r["url"], "status": r["status"], "mime": r.get("mimeType"),
                                          "protocol": r.get("protocol")})
                    elif m.get("method") in ("Runtime.consoleAPICalled", "Runtime.exceptionThrown"):
                        cdp.events.append(m)
            del orig_reader
            rd = asyncio.create_task(reader())
            await cdp.call("Page.enable")
            await cdp.call("Runtime.enable")
            await cdp.call("Network.enable")
            t0 = time.perf_counter()
            await cdp.call("Page.navigate", url=base + "/")
            for _ in range(600):  # up to 120 s for a cold start
                st = await cdp.js("document.readyState + '|' + !!document.getElementById('ask')")
                if st == "complete|true":
                    break
                await asyncio.sleep(0.2)
            report["checks"]["page_load_s"] = round(time.perf_counter() - t0, 2)
            report["checks"]["page_title"] = await cdp.js("document.title")
            report["checks"]["location"] = await cdp.js("location.href")
            report["checks"]["stylesheet_applied"] = await cdp.js(
                "getComputedStyle(document.getElementById('send')).cursor + '|' + document.styleSheets.length")

            tq = time.perf_counter()
            s1 = await ask(cdp, q1, timeout=240)
            (out / "browser_turn1_samples.json").write_text(json.dumps(s1, indent=1))
            streaming_lens = [s["len"] for s in s1 if s["streaming"]]
            report["checks"]["turn1"] = {
                "query": q1, "final": s1[-1],
                "distinct_answer_lengths_while_streaming": len({l for l in streaming_lens if l > 0}),
                "first_text_at_s": next((s["t"] for s in s1 if s["len"] > 0), None),
                "streaming_ended_at_s": next((s["t"] for s in s1 if s["streaming"] is False), None),
                "send_disabled_during_stream": any(s["sendDisabled"] for s in s1 if s["streaming"]),
                "wall_s": round(time.perf_counter() - tq, 2),
                "answer_text": await cdp.js("document.querySelector('#turns .turn:last-child .answer').textContent"),
                "cite_hrefs": await cdp.js("[...document.querySelectorAll('#turns .turn:last-child .answer a.cite')].map(a => a.textContent + ' ' + a.getAttribute('href'))"),
            }
            await shot(cdp, out / "browser_turn1_answer.png")
            report["checks"]["routing_rows"] = await cdp.js("[...document.querySelectorAll('#routing-rows tr')].map(r => r.textContent.slice(0, 220))")
            await cdp.js("document.querySelector('nav button[data-tab=trace]').click(); true")
            report["checks"]["trace_items"] = await cdp.js("[...document.querySelectorAll('#trace li')].map(r => r.textContent.slice(0, 200))")
            await shot(cdp, out / "browser_turn1_trace.png")
            first = await cdp.js("(() => { const a = document.querySelector('#turns .turn:last-child .answer a.cite'); if (!a) return null; a.click(); return a.textContent; })()")
            await asyncio.sleep(2.0)
            viewer = await cdp.js("({title: document.getElementById('viewer-title').textContent, meta: document.getElementById('viewer-meta').textContent, text: document.getElementById('viewer-text').textContent, link: document.getElementById('viewer-link').href, hidden: document.getElementById('viewer').hidden, tab: document.querySelector('nav button[aria-selected=true]').dataset.tab})")
            chunk_id = viewer["meta"].split("chunk ")[-1].strip() if viewer else None
            api = json.load(urllib.request.urlopen(base + "/api/source/chunk/" + urllib.parse.quote(chunk_id, safe=""))) if chunk_id else {}
            report["checks"]["citation_click"] = {
                "clicked": first, "viewer_title": viewer["title"], "viewer_meta": viewer["meta"], "viewer_tab": viewer["tab"],
                "viewer_hidden": viewer["hidden"], "viewer_text_equals_api": bool(api) and viewer["text"] == api.get("text"),
                "viewer_text_head": viewer["text"][:400], "viewer_link": viewer["link"],
                "link_equals_anchor": viewer["link"] == api.get("html_anchor_url"), "chunk_id": chunk_id,
                "api_doc_id": api.get("doc_id"),
            }
            await shot(cdp, out / "browser_turn1_source_viewer.png")

            s2 = await ask(cdp, q2, timeout=240)
            (out / "browser_turn2_samples.json").write_text(json.dumps(s2, indent=1))
            report["checks"]["turn2"] = {
                "query": q2, "final": s2[-1],
                "answer_text": await cdp.js("document.querySelector('#turns .turn:last-child .answer').textContent"),
                "cite_links": await cdp.js("document.querySelectorAll('#turns .turn:last-child .answer a.cite').length"),
                "status": await cdp.js("[...document.querySelectorAll('#turns .turn:last-child .status')].map(s => s.textContent)"),
            }
            await cdp.js("document.querySelector('nav button[data-tab=routing]').click(); true")
            report["checks"]["routing_rows_after_turn2"] = await cdp.js("[...document.querySelectorAll('#routing-rows tr')].map(r => r.textContent.slice(0, 220))")
            await shot(cdp, out / "browser_turn2_routing.png")
            report["responses"] = [r for r in responses if r["url"].startswith(base)]
            report["non_origin_responses"] = sorted({urllib.parse.urlsplit(r["url"]).netloc for r in responses
                                                     if not r["url"].startswith(base) and not r["url"].startswith("data:")})
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
    print(json.dumps({k: v for k, v in report["checks"].items() if k not in ("turn1", "turn2")}, indent=1, ensure_ascii=False)[:5000])
    for k in ("turn1", "turn2"):
        t = dict(report["checks"].get(k, {}))
        t["answer_text"] = (t.get("answer_text") or "")[:700]
        print(k, json.dumps(t, indent=1, ensure_ascii=False))
    print("responses", json.dumps(report.get("responses"), indent=0)[:1500], "non_origin", report.get("non_origin_responses"))
    print("console_errors", len(report.get("console_errors", [])))


if __name__ == "__main__":
    asyncio.run(main())
