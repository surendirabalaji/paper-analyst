"""Phase 8 browser check: real headless Chrome (CDP) against a running app (local or deployed).

1. Error state with no LLM spend: opens the page with a crafted #fault=...&fault_token=<wrong> link. The token
   must be cleared from the address bar (P7-N3), and the 403 must settle the turn on a visible error rather
   than "Planning…" (P7-N2).
2. A corpus + database question: tokens must arrive before `done`. Checks the per-turn routing chips and the
   Routing tab, the grouped Trace (per turn and agent, with status badges), Markdown rendered as elements
   (no raw ** / ---), and the C# and D# viewers (paper title, not the chunk id: P2-N3).
3. A Unicode question (accents, CJK, emoji) that needs live web results: the question and answer must render
   intact, and W# must open the full snippet (P5-N4).
4. Dark mode: contrast of the viewer link and citation links (P2-N5). Screenshots are taken throughout.

REAL provider and Tavily calls (2 questions). Usage:
  python tests/e2e/phase8_browser_ui.py BASE_URL OUT_DIR CHROME_PROFILE_DIR
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
from phase2_browser_cdp import CDP, CHROME, shot  # noqa: E402

PORT = 9338
Q_CORPUS = "How many papers in the corpus were published in 2025, and how does Reflexion use verbal self-reflection?"
Q_UNICODE = "Quelle est la dernière version de LangGraph publiée ? 另外，请用一句中文总结 ReAct 的核心思想 😀"

# Records every change of the last answer's text length with a timestamp, and when the status line first
# shows the final "TTFT … total …" summary (written by the `done` handler).
OBSERVE = """(() => {
  window.__obs = {changes: [], doneAt: null, streamingSeen: false};
  const mo = new MutationObserver(() => {
    const t = [...document.querySelectorAll('#turns .turn')].pop();
    if (!t) return;
    const a = t.querySelector('.answer'), s = t.querySelector('.status');
    const n = a ? a.textContent.length : 0, last = window.__obs.changes.at(-1);
    if (a && a.classList.contains('streaming') && n > 0) window.__obs.streamingSeen = true;
    if (n && (!last || last.n !== n)) window.__obs.changes.push({t: performance.now(), n});
    if (!window.__obs.doneAt && s && /total/.test(s.textContent)) window.__obs.doneAt = performance.now();
  });
  mo.observe(document.getElementById('turns'), {subtree: true, childList: true, characterData: true});
  return true;
})()"""

TURN = """(() => {
  const t = [...document.querySelectorAll('#turns .turn')].pop();
  const a = t.querySelector('.answer');
  const links = [...a.querySelectorAll('a.cite')].map(x => x.textContent);
  return {
    question: t.querySelector('.question').textContent,
    answer_text: a.textContent,
    answer_len: a.textContent.length,
    status: t.querySelector('.status').textContent,
    status_error: t.querySelector('.status').classList.contains('error'),
    turn_badge: t.querySelector('.turn-bar .badge').textContent,
    routing_line: t.querySelector('.routing-line').textContent,
    routing_chips: [...t.querySelectorAll('.routing-line .chip')].map(c => c.textContent),
    links, kinds: [...new Set(links.map(x => x[0]))],
    md: {strong: a.querySelectorAll('strong').length, li: a.querySelectorAll('li').length,
         headings: a.querySelectorAll('h3,h4,h5,h6').length, hr: a.querySelectorAll('hr').length,
         code: a.querySelectorAll('code').length, p: a.querySelectorAll('p').length},
    raw_markdown_left: /\\*\\*|^\\s*---\\s*$|^#{1,6}\\s/m.test(a.textContent),
    unsafe_nodes: document.querySelectorAll('#turns script, #turns img, #turns iframe, #turns [onerror], #turns [onclick]').length,
  };
})()"""

TRACE = """(() => {
  const d = [...document.querySelectorAll('#trace details.trace-turn')].pop();
  return {
    turn_groups: document.querySelectorAll('#trace details.trace-turn').length,
    turn_summary: d.querySelector('summary').textContent,
    flow: d.querySelector('.flow').textContent,
    agents: [...d.querySelectorAll('details.trace-agent')].map(g => ({
      name: g.querySelector('.agent-name').textContent,
      badges: [...g.querySelectorAll(':scope > summary .badge')].map(b => b.textContent),
      rows: g.querySelectorAll('li').length})),
    transitions: [...d.querySelectorAll('li')].map(l => l.textContent).filter(x => /Supervisor|Request received/.test(x)).slice(0, 8),
    rows_total: d.querySelectorAll('li').length,
    collapsible: d.tagName === 'DETAILS' && !!d.querySelector('details.trace-agent > summary'),
  };
})()"""

VIEWER = """({kind: document.getElementById('viewer-kind').textContent,
  title: document.getElementById('viewer-title').textContent,
  sub: document.getElementById('viewer-sub').textContent,
  meta: document.getElementById('viewer-meta').textContent,
  text_len: document.getElementById('viewer-text').textContent.length,
  text_head: document.getElementById('viewer-text').textContent.slice(0, 160),
  scrollable: getComputedStyle(document.getElementById('viewer-text')).overflowY,
  link: document.getElementById('viewer-link').href,
  tab: document.querySelector('nav button[aria-selected=true]').dataset.tab})"""

CONTRAST = """(() => {
  const rgb = (c) => c.match(/[\\d.]+/g).slice(0, 3).map(Number);
  const lum = ([r, g, b]) => { const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b); };
  const ratio = (a, b) => { const [x, y] = [lum(rgb(a)), lum(rgb(b))].sort((p, q) => q - p); return +((x + 0.05) / (y + 0.05)).toFixed(2); };
  const bg = getComputedStyle(document.getElementById('panel')).backgroundColor;
  const cite = document.querySelector('a.cite');
  return {dark: matchMedia('(prefers-color-scheme: dark)').matches, panel_bg: bg,
          viewer_link: ratio(getComputedStyle(document.getElementById('viewer-link')).color, bg),
          cite_link: cite ? ratio(getComputedStyle(cite).color, getComputedStyle(document.getElementById('chat')).backgroundColor) : null,
          body_text: ratio(getComputedStyle(document.body).color, getComputedStyle(document.body).backgroundColor)};
})()"""


async def wait_idle(cdp, timeout):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        s = await cdp.js("(() => { const t = [...document.querySelectorAll('#turns .turn')].pop();"
                         " return !!t && !t.querySelector('.answer').classList.contains('streaming')"
                         " && !document.getElementById('send').disabled; })()")
        if s:
            return round(time.perf_counter() - t0, 1)
        await asyncio.sleep(0.25)
    return None


async def submit(cdp, q):
    await cdp.js(OBSERVE)
    await cdp.js(f"document.getElementById('q').value = {json.dumps(q)}; document.getElementById('ask').requestSubmit(); true")


async def click_cite(cdp, prefix):
    return await cdp.js(f"(() => {{ const a = [...document.querySelectorAll('#turns .turn:last-child .answer a.cite')]"
                        f".find(x => x.textContent.startsWith({json.dumps(prefix)})); if (!a) return null; a.click(); return a.textContent; }})()")


async def tab(cdp, name):
    await cdp.js(f"document.querySelector('nav button[data-tab={name}]').click(); true")
    await asyncio.sleep(0.3)


async def main():
    base, out, profile = sys.argv[1].rstrip("/"), pathlib.Path(sys.argv[2]), sys.argv[3]
    out.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
                             "--no-first-run", "--no-default-browser-check", "--window-size=1500,1100", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rep = {"base": base, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checks": {}}
    C = rep["checks"]
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
            # light theme first (deterministic, whatever the OS uses); step 4 switches to dark
            await cdp.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "light"}])

            async def load(url):
                await cdp.call("Page.navigate", url=url)
                for _ in range(300):
                    if await cdp.js("document.readyState + '|' + !!document.getElementById('ask')") == "complete|true":
                        return
                    await asyncio.sleep(0.2)

            # ---- 1. error state + fault-token hygiene (the server refuses before any LLM call)
            await load(base + "/#fault=provider:*:down&fault_token=not-the-real-token-0123456789")
            await asyncio.sleep(0.5)
            rep["fault_link"] = await cdp.js("({hash: location.hash, href_has_token: location.href.includes('fault_token'),"
                                             " banner: (document.querySelector('.banner.fault') || {}).textContent || null,"
                                             " single_turn: document.getElementById('single-turn').textContent})")
            await submit(cdp, "test question for the error state")
            rep["error_wait_s"] = await wait_idle(cdp, 30)
            rep["error_turn"] = await cdp.js(TURN)
            await shot(cdp, out / "01_http_error_state.png")
            et = rep["error_turn"]
            C["P7-N3 token cleared from URL"] = rep["fault_link"]["hash"] == "" and not rep["fault_link"]["href_has_token"]
            C["P7-N2 HTTP error settles the turn"] = (et["status_error"] and "HTTP 403" in et["status"]
                                                      and "Planning" not in et["status"] and et["turn_badge"] == "failed")
            C["single-turn banner visible"] = "no conversation memory" in rep["fault_link"]["single_turn"]

            # ---- 2. corpus + database question
            await load(base + "/")
            await submit(cdp, Q_CORPUS)
            rep["q1_wait_s"] = await wait_idle(cdp, 240)
            rep["q1_obs"] = await cdp.js("window.__obs")
            rep["q1"] = await cdp.js(TURN)
            obs = rep["q1_obs"]
            before = [c for c in obs["changes"] if obs["doneAt"] and c["t"] < obs["doneAt"]]
            rep["q1_stream"] = {"length_changes": len(obs["changes"]), "changes_before_done": len(before),
                                "first_token_to_done_ms": round(obs["doneAt"] - obs["changes"][0]["t"]) if obs["doneAt"] and obs["changes"] else None,
                                "streaming_class_seen": obs["streamingSeen"]}
            await shot(cdp, out / "02_corpus_answer_routing_line.png")
            q1 = rep["q1"]
            C["streaming: tokens before done"] = len(before) >= 3 and obs["streamingSeen"]
            C["per-turn routing line"] = q1["routing_line"].startswith("Routing:") and len(q1["routing_chips"]) >= 2
            C["markdown rendered, no raw ** / ---"] = not q1["raw_markdown_left"] and q1["md"]["p"] >= 1
            C["no unsafe nodes in turns"] = q1["unsafe_nodes"] == 0

            await tab(cdp, "routing")
            rep["routing_tab"] = await cdp.js(
                "[...document.querySelectorAll('#routing-rows tr')].map(r => r.textContent.slice(0, 240))")
            rep["routing_providers"] = await cdp.js("[...new Set([...document.querySelectorAll('#routing-rows .prov')].map(p => p.textContent))]")
            rep["routing_turn_row_span"] = await cdp.js(
                "(() => { const td = document.querySelector('#routing-rows tr.turn-head td');"
                " return td ? +(td.offsetWidth / document.querySelector('#tab-routing table').offsetWidth).toFixed(2) : 0; })()")
            await shot(cdp, out / "03_routing_tab.png")
            C["Routing tab turn row spans the table"] = rep["routing_turn_row_span"] > 0.95
            C["Routing tab rows with provider/model/rationale"] = len(rep["routing_tab"]) >= 3 and all(
                "·" in r for r in rep["routing_tab"][1:])

            await tab(cdp, "trace")
            rep["q1_trace"] = await cdp.js(TRACE)
            await shot(cdp, out / "04_trace_grouped.png")
            tr = rep["q1_trace"]
            names = [a["name"] for a in tr["agents"]]
            C["trace grouped per turn and agent"] = tr["collapsible"] and len(tr["agents"]) >= 3 and names[0].startswith("Orchestration")
            C["trace status badges"] = all(a["badges"] and a["badges"][0] in ("done", "partial", "failed", "skipped", "ended")
                                           for a in tr["agents"])
            C["readable transitions"] = any(t.startswith("+") and "Supervisor hands task" in t for t in tr["transitions"])

            got = await click_cite(cdp, "C")
            await asyncio.sleep(1.0)
            rep["c_viewer"] = {"clicked": got, **(await cdp.js(VIEWER))} if got else None
            await shot(cdp, out / "05_viewer_paper_chunk.png")
            v = rep["c_viewer"]
            C["C# opens viewer with paper title (P2-N3)"] = bool(v) and v["tab"] == "sources" and v["kind"] == "paper" and \
                "::" not in v["title"] and "arxiv.org" in v["link"] and v["text_len"] > 200 and "Loading" not in v["text_head"]
            got = await click_cite(cdp, "D")
            await asyncio.sleep(0.5)
            rep["d_viewer"] = {"clicked": got, **(await cdp.js(VIEWER))} if got else None
            if got:
                await shot(cdp, out / "06_viewer_database_row.png")
            v = rep["d_viewer"]
            C["D# opens SQL viewer"] = bool(v) and v["kind"] == "database" and v["text_head"].startswith("SQL:")

            # ---- 3. Unicode + live web question (same page: second turn)
            await submit(cdp, Q_UNICODE)
            rep["q2_wait_s"] = await wait_idle(cdp, 240)
            rep["q2"] = await cdp.js(TURN)
            obs2 = await cdp.js("window.__obs")
            rep["q2_stream"] = {"changes_before_done": len([c for c in obs2["changes"] if obs2["doneAt"] and c["t"] < obs2["doneAt"]])}
            await shot(cdp, out / "07_unicode_turn.png")
            q2 = rep["q2"]
            non_ascii = sorted({ch for ch in q2["answer_text"] if ord(ch) > 127})
            rep["q2_non_ascii_chars"] = "".join(non_ascii[:80])
            C["Unicode question rendered intact"] = q2["question"] == Q_UNICODE
            C["Unicode answer streamed (non-ASCII, no U+FFFD)"] = bool(non_ascii) and "�" not in q2["answer_text"] \
                and any("一" <= ch <= "鿿" for ch in non_ascii)
            got = await click_cite(cdp, "W")
            await asyncio.sleep(0.5)
            rep["w_viewer"] = {"clicked": got, **(await cdp.js(VIEWER))} if got else None
            if got:
                await shot(cdp, out / "08_viewer_web_full_snippet.png")
            v = rep["w_viewer"]
            C["W# opens viewer with full snippet (P5-N4)"] = bool(v) and v["kind"] == "web" and v["scrollable"] == "auto" \
                and (v["text_len"] > 280 or "characters" in v["meta"])
            await tab(cdp, "trace")
            rep["q2_trace"] = await cdp.js(TRACE)
            rep["mcp_rows"] = await cdp.js("[...[...document.querySelectorAll('#trace details.trace-turn')].pop()"
                                           ".querySelectorAll('li')].map(l => l.textContent).filter(t => t.includes('⇄ MCP')).slice(0, 4)")
            await shot(cdp, out / "09_trace_unicode_turn_mcp.png")
            C["two turn groups, older folded"] = rep["q2_trace"]["turn_groups"] == 2 and await cdp.js(
                "!document.querySelectorAll('#trace details.trace-turn')[0].open")

            # ---- 4. dark mode contrast
            await cdp.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "dark"}])
            await click_cite(cdp, "C") or await click_cite(cdp, "W")
            await asyncio.sleep(1.0)
            rep["dark_contrast"] = await cdp.js(CONTRAST)
            await shot(cdp, out / "10_dark_mode_viewer.png")
            await tab(cdp, "trace")
            await shot(cdp, out / "11_dark_mode_trace.png")
            dc = rep["dark_contrast"]
            C["dark-mode link contrast ≥ 4.5 (P2-N5)"] = dc["dark"] and dc["viewer_link"] >= 4.5 and (dc["cite_link"] or 0) >= 4.5

            rep["console_errors"] = [e for e in cdp.events if e.get("method") == "Runtime.exceptionThrown"
                                     or e.get("params", {}).get("type") == "error"]
            C["no console errors"] = not rep["console_errors"]
            rd.cancel()
    finally:
        proc.terminate()
    rep["passed"] = sum(bool(v) for v in C.values())
    rep["total"] = len(C)
    (out / "browser_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({"checks": C, "passed": f"{rep['passed']}/{rep['total']}", "q1_stream": rep.get("q1_stream"),
                      "q1_links": rep.get("q1", {}).get("links"), "q2_links": rep.get("q2", {}).get("links"),
                      "dark": rep.get("dark_contrast")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
