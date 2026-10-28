"""Tester Phase 8: REAL deployed app, REAL headless Chrome (CDP), REAL providers / Tavily / papers.db.

A fetch() tee (response.clone()) records every SSE event with its client arrival time, so each check compares
what the page shows with what the server actually sent:
  - streaming: answer DOM grows over time before `done` arrives; client arrival spread vs server t_ms spread
  - routing: one Routing-tab row and chip per llm_route/fallback, same provider label and model id; returned model
  - trace: one group per turn, agent groups with status badges, transitions as sentences
  - citations: every linked id opens the right viewer; C# text == /api/source/chunk, W# == the event snippet,
    D# SQL/rows == the event and an independent local read-only SQL run
  - two tabs at once: different request ids, no foreign citations / routing rows / trace text
  - Unicode (RTL/CJK/emoji) question, a stateless follow-up, an over-length query (HTTP 422, no spend) and a
    mid-stream network drop (CDP offline) on the deployed service.

Usage: python tests/e2e/phase8_tester_deployed.py BASE_URL OUT_DIR CHROME_PROFILE_DIR
"""
import asyncio
import json
import pathlib
import sqlite3
import subprocess
import sys
import time
import urllib.parse
import urllib.request

import websockets

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from phase2_browser_cdp import CDP, CHROME, shot  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
PORT = 9343
Q_TOT = "What lets the Tree of Thoughts approach back out of a dead end and try a different line of reasoning?"
Q_DATA = "List the corpus papers whose titles mention debate, then give a short summary of what they argue."
Q_RTL = "ما هو أحدث إصدار من مكتبة LangGraph؟ Please answer in English and mention the release date 🤖📦"
Q_FOLLOW = "And what did it say about its limitations?"
Q_DROP = "Briefly, how does MetaGPT assign roles to its agents?"

TAP = r"""(() => {
  window.__tap = {events: [], chunks: 0, t0: performance.now()};
  window.__obs = {changes: []};
  const of = window.fetch.__orig || window.fetch;
  window.fetch = async (...a) => {
    const r = await of(...a);
    if (String(a[0]).includes('/api/chat') && r.ok) {
      const c = r.clone();
      (async () => { const rd = c.body.getReader(); const dec = new TextDecoder(); let buf = '';
        try { for (;;) { const {done, value} = await rd.read(); if (done) break; window.__tap.chunks++;
          buf += dec.decode(value, {stream: true}); let i;
          while ((i = buf.indexOf('\n\n')) >= 0) { const f = buf.slice(0, i); buf = buf.slice(i + 2);
            const m = f.match(/^event: (.*)$/m), d = f.match(/^data: (.*)$/m);
            if (m && d) window.__tap.events.push({ev: m[1], t: performance.now(), d: JSON.parse(d[1])}); } } }
        catch (e) { window.__tap.err = String(e); } })();
    }
    return r; };
  window.fetch.__orig = of;
  const mo = new MutationObserver(() => { const t = [...document.querySelectorAll('#turns .turn')].pop(); if (!t) return;
    const a = t.querySelector('.answer'); const n = a.textContent.length; const l = window.__obs.changes.at(-1);
    if (n && (!l || l.n !== n)) window.__obs.changes.push({t: performance.now(), n, streaming: a.classList.contains('streaming')}); });
  mo.observe(document.getElementById('turns'), {subtree: true, childList: true, characterData: true});
  return true; })()"""

STATE = r"""(() => { const t = [...document.querySelectorAll('#turns .turn')].pop(); const a = t.querySelector('.answer');
  const d = [...document.querySelectorAll('#trace details.trace-turn')].pop();
  const rows = [...document.querySelectorAll('#routing-rows tr')];
  return {question: t.querySelector('.question').textContent, answer: a.textContent, streaming: a.classList.contains('streaming'),
    status: t.querySelector('.status').textContent, status_cls: t.querySelector('.status').className,
    badge: t.querySelector('.turn-bar .badge').textContent, send_disabled: document.getElementById('send').disabled,
    links: [...a.querySelectorAll('a.cite')].map(x => x.textContent), ext_links: [...a.querySelectorAll('a.ext')].map(x => x.href),
    chips: [...t.querySelectorAll('.routing-line .chip')].map(c => ({text: c.textContent, model: c.title, cls: c.className})),
    md: {strong: a.querySelectorAll('strong').length, li: a.querySelectorAll('li').length, p: a.querySelectorAll('p').length,
         h: a.querySelectorAll('h3,h4,h5,h6').length, table: a.querySelectorAll('table').length},
    raw_md: /\*\*|^#{1,6}\s|^\s*---\s*$/m.test(a.textContent),
    trace: {turns: document.querySelectorAll('#trace details.trace-turn').length, open: d.open,
      summary: d.querySelector('summary').textContent, flow: d.querySelector('.flow').textContent,
      groups: [...d.querySelectorAll('details.trace-agent')].map(g => ({name: g.querySelector('.agent-name').textContent,
        badges: [...g.querySelectorAll(':scope > summary .badge')].map(b => b.textContent), n: g.querySelectorAll('li').length})),
      rows: [...d.querySelectorAll('li')].map(li => li.textContent)},
    routing_rows: rows.map(r => ({cls: r.className, cells: [...r.children].map(c => c.textContent)})),
    sources: [...document.querySelectorAll('#sources li')].map(li => li.textContent),
    xss: window.__xss || 0,
    unsafe: document.querySelectorAll('#turns script, #turns img, #turns iframe, #turns svg').length}; })()"""

VIEWER = """({kind: document.getElementById('viewer-kind').textContent, title: document.getElementById('viewer-title').textContent,
  sub: document.getElementById('viewer-sub').textContent, meta: document.getElementById('viewer-meta').textContent,
  text: document.getElementById('viewer-text').textContent, link: document.getElementById('viewer-link').href,
  link_hidden: document.getElementById('viewer-link').hidden, link2: document.getElementById('viewer-link2').href,
  tab: document.querySelector('nav button[aria-selected=true]').dataset.tab})"""


async def open_tab(url):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/json/new?{urllib.parse.quote(url, safe='')}", method="PUT")
    t = json.load(urllib.request.urlopen(req))
    ws = await websockets.connect(t["webSocketDebuggerUrl"], max_size=None)
    cdp = CDP(ws)
    cdp._rd = asyncio.create_task(cdp.reader())
    await cdp.call("Page.enable")
    await cdp.call("Runtime.enable")
    await cdp.call("Page.navigate", url=url)
    await asyncio.sleep(2.0)
    await cdp.js(TAP)
    await cdp.js("window.__xss = 0; true")
    return cdp


async def submit(cdp, q):
    await cdp.js(TAP)
    await cdp.js(f"document.getElementById('q').value = {json.dumps(q)}; document.getElementById('ask').requestSubmit(); true")


async def wait_done(cdp, timeout=170):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        await asyncio.sleep(0.5)
        s = await cdp.js("({s: [...document.querySelectorAll('#turns .turn')].pop().querySelector('.answer').classList.contains('streaming'), d: document.getElementById('send').disabled})")
        if not s["s"] and not s["d"]:
            break
    await asyncio.sleep(0.5)
    return await cdp.js(STATE), await cdp.js("window.__tap"), await cdp.js("window.__obs")


def streaming_metrics(tap, obs):
    evs = tap["events"]
    toks = [e for e in evs if e["ev"] == "token"]
    done = next((e for e in evs if e["ev"] == "done"), None)
    ch = obs["changes"]
    before_done = [c for c in ch if done and c["t"] < done["t"]]
    return {
        "n_events": len(evs), "n_token_events": len(toks), "network_chunks": tap["chunks"],
        "client_first_token_to_done_s": round((done["t"] - toks[0]["t"]) / 1000, 2) if toks and done else None,
        "server_first_token_to_done_s": round((done["d"]["t_ms"] - toks[0]["d"]["t_ms"]) / 1000, 2) if toks and done else None,
        "client_token_arrival_span_s": round((toks[-1]["t"] - toks[0]["t"]) / 1000, 2) if toks else None,
        "server_token_t_ms_span_s": round((toks[-1]["d"]["t_ms"] - toks[0]["d"]["t_ms"]) / 1000, 2) if toks else None,
        "dom_length_changes_before_done": len(before_done), "dom_changes_total": len(ch),
        "dom_first_text_before_done_s": round((done["t"] - before_done[0]["t"]) / 1000, 2) if before_done else None,
        "server_ttft_ms": done["d"]["timings"].get("ttft_ms") if done else None,
        "request_id": evs[0]["d"].get("request_id") if evs else None,
        "request_ids": sorted({e["d"].get("request_id") for e in evs if e["d"].get("request_id")}),
    }


def routing_checks(state, tap):
    routes = [e["d"] for e in tap["events"] if e["ev"] in ("llm_route", "fallback")]
    results = {e["d"]["decision_id"]: e["d"] for e in tap["events"] if e["ev"] == "llm_result"}
    turn_no = None
    rows = state["routing_rows"]
    # the rows for the last turn are those after the last turn-head row
    idx = max(i for i, r in enumerate(rows) if r["cls"] == "turn-head")
    mine = rows[idx + 1:]
    pairs = []
    for d, r in zip(routes, mine):
        res = results.get(d["decision_id"], {})
        pairs.append({"agent_call": f'{d["agent"]} / {d["call_site"]}', "row_agent": r["cells"][1], "provider_label": d["provider_label"],
                      "model": d["model_id"], "row_provider": r["cells"][2], "model_returned": res.get("model_returned"),
                      "upstream": res.get("upstream_provider"), "status": res.get("status"),
                      "row_ok": r["cells"][1] == f'{d["agent"]} / {d["call_site"]}' and d["model_id"] in r["cells"][2]
                      and d["provider_label"] in r["cells"][2] and d["rationale"] in r["cells"][3],
                      "rationale": d["rationale"][:160], "dimension": d["dimension"]})
    chip_total = sum(int(c["text"].split("×")[1].split()[0]) if "×" in c["text"] else 1 for c in state["chips"])
    return {"n_routes": len(routes), "n_rows": len(mine), "pairs": pairs, "all_rows_match": all(p["row_ok"] for p in pairs) and len(routes) == len(mine),
            "chip_total": chip_total, "chips_match": chip_total == len(routes),
            "providers": sorted({d["provider"] for d in routes}), "turn_head": rows[idx]["cells"][0], "turn": turn_no}


async def check_citations(cdp, tap, base):
    cites = {e["d"]["cite_id"]: e["d"] for e in tap["events"] if e["ev"] == "citation"}
    webres = {r["cite_id"]: r for e in tap["events"] if e["ev"] == "web_search" for r in e["d"].get("results", [])}
    out = []
    for cid, d in cites.items():
        ok = await cdp.js(f"""(() => {{ const a = [...[...document.querySelectorAll('#turns .turn')].pop().querySelectorAll('.answer a.cite')].find(x => x.textContent === {json.dumps(cid)});
            if (!a) return false; a.click(); return true; }})()""")
        await asyncio.sleep(1.5)
        v = await cdp.js(VIEWER)
        rec = {"cite": cid, "type": d["type"], "linked_in_answer": ok, "viewer_kind": v["kind"], "viewer_title": v["title"][:120]}
        if d["type"] == "paper_chunk":
            api = json.load(urllib.request.urlopen(base + "/api/source/chunk/" + urllib.parse.quote(d["chunk_id"], safe="")))
            rec.update(chunk_id=d["chunk_id"], text_equals_api=v["text"] == api["text"], title_is_paper=api["title"] in v["title"],
                       not_chunk_id_title=d["chunk_id"] not in v["title"], section=v["sub"][:80], link=v["link"],
                       link_is_anchor=v["link"] == api.get("html_anchor_url"), abs=v["link2"])
            rec["ok"] = ok and v["kind"] == "paper" and rec["text_equals_api"] and rec["title_is_paper"] and rec["link_is_anchor"]
        elif d["type"] == "web":
            snip = d.get("snippet") or ""
            rec.update(url=d["source_url"], snippet_len=len(snip), viewer_len=len(v["text"]), text_equals_event=v["text"] == snip,
                       equals_web_search_result=snip == webres.get(cid, {}).get("snippet"), link_ok=v["link"] == d["source_url"],
                       retrieved=d.get("retrieved_at"), longer_than_280=len(snip) > 280)
            rec["ok"] = ok and v["kind"] == "web" and rec["text_equals_event"] and rec["link_ok"] and rec["equals_web_search_result"]
        else:
            rec.update(sql=d.get("sql"), rows=d.get("rows"), viewer_has_sql=(d.get("sql") or "?") in v["text"])
            try:
                con = sqlite3.connect(f"file:{ROOT / 'data' / 'papers.db'}?mode=ro", uri=True)
                truth = [list(r) for r in con.execute(d["sql"]).fetchall()]
                con.close()
            except Exception as e:  # noqa: BLE001
                truth = f"error: {e}"
            rec["independent_sql_rows"] = truth
            rec["row_in_truth"] = isinstance(truth, list) and all(r in truth for r in (d.get("rows") or []))
            rec["ok"] = ok and v["kind"] == "database" and rec["viewer_has_sql"] and rec["row_in_truth"]
        out.append(rec)
    return out


async def main():
    base, out, profile = sys.argv[1].rstrip("/"), pathlib.Path(sys.argv[2]), sys.argv[3]
    out.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
                             "--no-first-run", "--no-default-browser-check", "--window-size=1500,1100", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rep = {"base": base, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "real": True}
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/version")
                break
            except Exception:
                time.sleep(0.2)
        A = await open_tab(base + "/")
        B = await open_tab(base + "/")
        for c in (A, B):
            await c.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "light"}])

        # ---- 0. over-length query: HTTP 422 from the real server, no LLM spend
        await A.js("document.getElementById('q').value = 'x'.repeat(2500); document.getElementById('ask').requestSubmit(); true")
        await asyncio.sleep(3)
        rep["http422"] = await A.js(STATE)

        # ---- 1. two tabs at once: corpus question (A) and Data -> Summary question (B)
        await submit(A, Q_TOT)
        await asyncio.sleep(0.3)
        await submit(B, Q_DATA)
        (sA, tA, oA), (sB, tB, oB) = await asyncio.gather(wait_done(A), wait_done(B))
        await shot(A, out / "d1_tab_a_answer.png")
        await shot(B, out / "d2_tab_b_answer.png")
        rep["A"] = {"state": sA, "stream": streaming_metrics(tA, oA), "routing": routing_checks(sA, tA), "events": tA["events"]}
        rep["B"] = {"state": sB, "stream": streaming_metrics(tB, oB), "routing": routing_checks(sB, tB), "events": tB["events"]}
        idsA = {e["d"]["cite_id"] for e in tA["events"] if e["ev"] == "citation"}
        idsB = {e["d"]["cite_id"] for e in tB["events"] if e["ev"] == "citation"}
        rep["isolation"] = {
            "request_ids": [rep["A"]["stream"]["request_ids"], rep["B"]["stream"]["request_ids"]],
            "A_has_B_question": Q_DATA[:40] in json.dumps(sA, ensure_ascii=False),
            "B_has_A_question": Q_TOT[:40] in json.dumps(sB, ensure_ascii=False),
            "A_sources": len(sA["sources"]), "A_citations": len(idsA), "B_sources": len(sB["sources"]), "B_citations": len(idsB),
            "A_routing_turn_heads": [r["cells"][0] for r in sA["routing_rows"] if r["cls"] == "turn-head"],
            "B_routing_turn_heads": [r["cells"][0] for r in sB["routing_rows"] if r["cls"] == "turn-head"],
        }
        rep["A"]["citations"] = await check_citations(A, tA, base)
        await shot(A, out / "d1_tab_a_viewer.png")
        rep["B"]["citations"] = await check_citations(B, tB, base)
        await shot(B, out / "d2_tab_b_viewer.png")
        await B.js("document.querySelector('nav button[data-tab=trace]').click(); true")
        await shot(B, out / "d2_tab_b_trace.png")
        await A.js("document.querySelector('nav button[data-tab=routing]').click(); true")
        await shot(A, out / "d1_tab_a_routing.png")

        # ---- 2. stateless follow-up in tab A
        await submit(A, Q_FOLLOW)
        sF, tF, oF = await wait_done(A)
        rep["follow"] = {"state": {k: sF[k] for k in ("answer", "status", "badge", "links", "chips")},
                         "plan": [e["d"] for e in tF["events"] if e["ev"] == "plan"],
                         "routing": routing_checks(sF, tF)}

        # ---- 3. RTL + emoji + live web question in tab B, dark mode
        await B.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "dark"}])
        await submit(B, Q_RTL)
        sR, tR, oR = await wait_done(B)
        rep["rtl"] = {"state": sR, "stream": streaming_metrics(tR, oR), "routing": routing_checks(sR, tR),
                      "question_exact": sR["question"] == Q_RTL, "fffd": "�" in sR["answer"] + sR["question"],
                      "dir": await B.js("(() => { const t = [...document.querySelectorAll('#turns .turn')].pop(); return [t.querySelector('.question').dir, getComputedStyle(t.querySelector('.question')).direction]; })()"),
                      "events": tR["events"]}
        await shot(B, out / "d3_rtl_dark_answer.png")
        rep["rtl"]["citations"] = await check_citations(B, tR, base)
        await shot(B, out / "d3_rtl_dark_viewer.png")
        await B.js("document.querySelector('nav button[data-tab=trace]').click(); true")
        rep["rtl"]["trace_turns_open"] = await B.js("[...document.querySelectorAll('#trace details.trace-turn')].map(d => d.open)")
        await shot(B, out / "d3_rtl_dark_trace.png")

        # ---- 4. network drop mid-stream (CDP offline) in tab A
        await A.call("Network.enable")
        await submit(A, Q_DROP)
        t0 = time.perf_counter()
        dropped_at = None
        while time.perf_counter() - t0 < 150:
            await asyncio.sleep(0.25)
            n = await A.js("window.__tap.events.filter(e => e.ev === 'token').length")
            if n >= 3:
                await A.call("Network.emulateNetworkConditions", offline=True, latency=0, downloadThroughput=-1, uploadThroughput=-1)
                dropped_at = round(time.perf_counter() - t0, 2)
                break
        await asyncio.sleep(4)
        sD = await A.js(STATE)
        await A.call("Network.emulateNetworkConditions", offline=False, latency=0, downloadThroughput=-1, uploadThroughput=-1)
        rep["drop"] = {"dropped_at_s": dropped_at, "state": {k: sD[k] for k in ("answer", "status", "status_cls", "badge", "send_disabled", "streaming")},
                       "trace_last_rows": sD["trace"]["rows"][-3:]}
        await shot(A, out / "d4_network_drop.png")
        rep["console"] = [json.dumps(e)[:300] for c in (A, B) for e in c.events
                          if e.get("method") == "Runtime.exceptionThrown" or e["params"].get("type") == "error"]
        for c in (A, B):
            c._rd.cancel()
            await c.ws.close()
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except Exception:
            proc.kill()
        (out / "deployed_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    print("written", out / "deployed_report.json")


if __name__ == "__main__":
    asyncio.run(main())
