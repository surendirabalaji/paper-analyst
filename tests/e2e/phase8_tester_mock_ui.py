"""Tester Phase 8: the REAL app/static UI (index.html, md.js, app.js, styles.css, served unmodified) in REAL headless
Chrome, against a MOCK /api/chat that replays scripted SSE streams. No provider, Tavily or deployed calls.

What the mock lets us control that a live model cannot be made to produce on demand:
  xss       hostile Markdown and hostile values in every trace/routing/citation/viewer field
  split     Unicode (accents, CJK, emoji ZWJ, Arabic/Hebrew RTL) cut at arbitrary BYTE boundaries, CRLF framing,
            several frames per network chunk, keep-alive comments
  cut       the connection closes mid-answer without `done`
  http500   non-JSON 500 body; http422 FastAPI-style validation list
  errev     a non-recoverable `error` event followed by done(failed)
  partial   an injected worker failure, done(partial), dropped citation
  slow      a slow stream used to probe double submit (button and Enter key)
  isoA/isoB two turns that both define C1 with different sources (stale-registry check)
  long      ~3,000 deltas / ~20k characters (render cost, final structure)
  stall     headers + one ping, then silence (60 s client watchdog)

Usage: python tests/e2e/phase8_tester_mock_ui.py OUT_DIR CHROME_PROFILE_DIR [--with-stall]
"""
import asyncio
import json
import pathlib
import subprocess
import sys
import threading
import time
import urllib.request

import uvicorn
import websockets
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from phase2_browser_cdp import CDP, CHROME, shot  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
STATIC = ROOT / "app" / "static"
APP_PORT, CDP_PORT = 8791, 9341
EVIL = '<img src=x onerror="window.__xss=1"><script>window.__xss=2</script><svg onload="window.__xss=3">'
UNI = "Résumé naïve — 检索增强生成 日本語 😀 👩‍🔬 🇫🇷 مرحبا بالعالم שלום עולם"

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/source/chunk/{chunk_id}")
async def source(chunk_id: str):
    if chunk_id == "missing::x::000":
        return JSONResponse({"detail": "unknown chunk id"}, 404)
    return {"chunk_id": chunk_id, "doc_id": "2210.03629v3", "title": f"T-{chunk_id} {EVIL}",
            "section_path": f"Sec {EVIL}", "published": "2022-10-06",
            "text": f"TEXT-OF[{chunk_id}] {EVIL} {UNI}", "html_anchor_url": "javascript:window.__xss=9",
            "abs_url": "https://arxiv.org/abs/2210.03629"}


def frame(ev, d):
    return f"event: {ev}\ndata: {json.dumps(d, ensure_ascii=False)}\n\n"


def chunk_ref(cid, chunk_id, title="ReAct", section="Method", href=None, src="https://arxiv.org/html/2210.03629v3#S3"):
    return {"cite_id": cid, "type": "paper_chunk", "doc_id": "2210.03629v3", "chunk_id": chunk_id, "title": title,
            "section": section, "score": 0.71, "href": href or f"/api/source/chunk/{urllib.request.quote(chunk_id, safe='')}",
            "source_url": src}


def base_events(rid, tag, chunks, extra=()):
    ev = [("meta", {"request_id": rid, "k": 6, "workers": ["research", "summary", "data"], "budget_s": 120, "t_ms": 1}),
          ("transition", {"from": "start", "to": "planner", "reason": "new request", "t_ms": 2}),
          ("llm_route", {"decision_id": f"{rid}-d1", "agent": "planner", "call_site": "plan", "provider": "gemini-openrouter",
                         "provider_label": "Gemini via OpenRouter", "model_id": "google/gemini-3.8-flash", "dimension": "cost",
                         "rationale": f"rationale-{tag}", "features": {"len": 10, "tag": tag}, "reasoning_effort": "low", "t_ms": 3}),
          ("llm_result", {"decision_id": f"{rid}-d1", "status": "ok", "attempt": 1, "input_tokens": 10, "output_tokens": 5,
                          "latency_ms": 100, "ttft_ms": 50, "cost_usd": 0.0001, "cost_source": "estimated", "t_ms": 4}),
          ("plan", {"plan_summary": f"plan-{tag}", "source": "planner", "notes": [],
                    "tasks": [{"id": "t1", "worker": "research", "needs_live": False, "instruction": f"instr-{tag}", "depends_on": []}],
                    "t_ms": 5}),
          ("transition", {"from": "supervisor", "to": "research", "task_id": "t1", "reason": "ready", "t_ms": 6}),
          ("retrieval", {"agent": "research", "tool": "corpus_search", "query": f"q-{tag}", "chunks": chunks, "retrieval_ms": 9, "t_ms": 7}),
          *extra,
          ("worker_result", {"agent": "research", "task_id": "t1", "status": "done", "attempt": 1,
                             "evidence_ids": [c["cite_id"] for c in chunks], "t_ms": 8}),
          ("transition", {"from": "supervisor", "to": "synthesize", "reason": "all tasks finished", "t_ms": 9}),
          ("llm_route", {"decision_id": f"{rid}-d2", "agent": "synthesizer", "call_site": "synthesize", "provider": "openai",
                         "provider_label": "OpenAI", "model_id": "gpt-6.1-sol", "dimension": "capability",
                         "rationale": f"synth-{tag}", "features": {"n_sources": len(chunks)}, "t_ms": 10})]
    return ev


def done(rid, status="done", dropped=(), unresolved=()):
    return ("done", {"status": status, "timings": {"ttft_ms": 1200, "total_ms": 3000, "retrieval_ms": 9},
                     "tokens_by_provider": {"openai": {"calls": 1, "input_tokens": 100, "output_tokens": 50}},
                     "cost_usd": 0.001, "cost_basis": {"estimated": 0}, "fallbacks": 0,
                     "unresolved_citations": list(unresolved), "dropped_citations": list(dropped),
                     "tasks": [{"id": "t1", "worker": "research", "status": status}], "faults_injected": [], "t_ms": 3000})


def tokens(text, n=8):
    return [("token", {"delta": text[i:i + n], "t_ms": 1000 + i}) for i in range(0, len(text), n)]


def citation(c, **kw):
    d = {"cite_id": c["cite_id"], "type": c["type"], "label": f'{c["title"]} — {c["section"]}', "title": c["title"],
         "section": c["section"], "doc_id": c["doc_id"], "chunk_id": c.get("chunk_id", ""), "href": c["href"],
         "source_url": c["source_url"], "retrieved_at": None, "found_by": "research"}
    d.update(kw)
    return ("citation", d)


XSS_MD = (f"## Heading {EVIL}\n\nPara {EVIL} with [C1] and [C9] and [Z1] and [C1, W1; D1].\n\n"
          "- [js link](javascript:window.__xss=4) [data](data:text/html,<script>window.__xss=5</script>)\n"
          "- [vb](vbscript:msgbox) [quote-break](https://ok.example/\"onmouseover=\"window.__xss=6) [ok](https://arxiv.org/abs/2210.03629)\n"
          "- ![image](https://evil.example/x.png) &lt;script&gt;window.__xss=7&lt;/script&gt; <a href=\"javascript:window.__xss=8\">a</a>\n\n"
          f"| col {EVIL} | b |\n|---|---|\n| <iframe src=javascript:window.__xss=10></iframe> | [D1] |\n\n"
          f"```html\n</pre><script>window.__xss=11</script>{EVIL}\n```\n\n> quote {EVIL} [W1]\n\n**bold {EVIL}** *em* `code {EVIL}`\n")


def scenario_events(name, rid):
    if name == "xss":
        c1 = chunk_ref("C1", "2210.03629v3::s3::001", title=f"Title {EVIL}", section=f"Sec {EVIL}")
        web = {"cite_id": "W1", "url": "javascript:window.__xss=12", "title": f"Web {EVIL}", "domain": f"evil{EVIL}",
               "score": 0.5, "retrieved_at": f"2026-10-28T00:00Z{EVIL}", "snippet": f"SNIP {EVIL} {UNI}"}
        sqlref = {"cite_id": "D1", "type": "sql_row", "title": f"Row {EVIL}", "href": "javascript:window.__xss=13",
                  "doc_id": "2210.03629v3", "rows": [[f"<b>{EVIL}</b>", 3]]}
        extra = [
            ("tool_call", {"agent": "research", "tool": f"tool{EVIL}", "args_safe": {"q": EVIL}, "t_ms": 7}),
            ("tool_result", {"agent": "research", "tool": f"tool{EVIL}", "status": "ok", "n_results": 1, "latency_ms": 5, "t_ms": 7}),
            ("web_search", {"agent": "research", "tool": "tavily_search", "server": f"srv{EVIL}", "purpose": EVIL, "query": EVIL,
                            "status": "ok", "results": [web], "latency_ms": 5, "t_ms": 7}),
            ("sql_result", {"agent": "data", "status": "ok", "sql": f"SELECT '{EVIL}'", "n_rows": 1, "truncated": False,
                            "elapsed_ms": 1, "cite_ids": ["D1"], "refs": [sqlref], "columns": [f"c{EVIL}", "n"], "t_ms": 7}),
            ("handoff", {"from_tasks": ["t0"], "to_task": "t1", "to_agent": "summary", "papers": [{"arxiv_id": EVIL, "cite_id": "D1"}],
                         "evidence_ids": [EVIL], "t_ms": 7}),
            ("error", {"scope": "worker", "agent": "research", "error_kind": EVIL, "message_safe": f"recoverable {EVIL}",
                       "recoverable": True, "t_ms": 7}),
            ("llm_route", {"decision_id": "x-d9", "agent": f"a{EVIL}", "call_site": EVIL, "provider": f"p\" onclick=\"window.__xss=14",
                           "provider_label": f"P{EVIL}", "model_id": EVIL, "dimension": EVIL, "rationale": EVIL,
                           "features": {EVIL: EVIL}, "t_ms": 7}),
            ("llm_result", {"decision_id": "x-d9", "status": "ok", "attempt": 1, "input_tokens": EVIL, "output_tokens": 1,
                            "latency_ms": 1, "upstream_provider": EVIL, "t_ms": 7}),
        ]
        ev = base_events(rid, "xss", [c1], extra)
        ev += tokens(XSS_MD, 5)
        ev += [citation(c1), ("citation", {"cite_id": "W1", "type": "web", "label": f"Web {EVIL}", "title": f"Web {EVIL}",
                                           "section": "", "snippet": web["snippet"], "doc_id": web["domain"], "chunk_id": "",
                                           "href": web["url"], "source_url": web["url"], "retrieved_at": web["retrieved_at"],
                                           "found_by": "research"}),
               ("citation", {"cite_id": "D1", "type": "sql_row", "label": f"Row {EVIL}", "title": f"Row {EVIL}",
                             "section": "corpus metadata (SQL)", "doc_id": "2210.03629v3", "chunk_id": "", "href": sqlref["href"],
                             "source_url": sqlref["href"], "found_by": "data", "sql": f"SELECT '{EVIL}'",
                             "columns": [f"c{EVIL}", "n"], "rows": sqlref["rows"]}),
               done(rid)]
        return ev
    if name in ("isoA", "isoB"):
        c1 = chunk_ref("C1", f"{name}::sec::000", title=f"Paper-{name}", section=f"Section-{name}")
        return base_events(rid, name, [c1]) + tokens(f"Answer {name} relies on [C1].") + [citation(c1), done(rid)]
    if name == "enc":  # chunk id that needs percent-encoding
        cid = "2305.10601v2::4 Experiments Game of 24 #1 ?a=b&c%::003"  # real ids are [A-Za-z0-9._:-] only; "/" is a separate info check
        c1 = chunk_ref("C1", cid, title="Tree of Thoughts", section="4 Experiments")
        c2 = chunk_ref("C2", "missing::x::000", title="Gone", section="Gone")
        return base_events(rid, "enc", [c1, c2]) + tokens("Encoded [C1]; missing [C2].") + [citation(c1), citation(c2), done(rid)]
    if name == "partial":
        c1 = chunk_ref("C1", "2303.11366v4::abstract::000", title="Reflexion", section="Abstract")
        c2 = chunk_ref("C2", "2303.11366v4::S4::001", title="Reflexion", section="Experiments")
        extra = [("fault_injected", {"kind": "worker", "name": "data", "agent": "data", "target": "worker:data:exception",
                                     "task_id": "t2", "note": "test fault", "t_ms": 7}),
                 ("worker_result", {"agent": "data", "task_id": "t2", "status": "failed", "attempt": 2, "error_kind": "worker_error",
                                    "reason": "injected test fault", "injected": True, "evidence_ids": [], "t_ms": 8})]
        return (base_events(rid, "partial", [c1, c2], extra) + tokens("Reflexion reflects [C1]. The count is unknown [C2].\n\n---\n**Note:** Data failed.")
                + [citation(c1), done(rid, "partial", dropped=["C2"])])
    if name == "errev":
        return base_events(rid, "errev", []) + [
            ("error", {"scope": "synthesize", "error_kind": "provider_unavailable", "message_safe": "all providers failed",
                       "recoverable": False, "t_ms": 20}), done(rid, "failed")]
    if name == "long":
        para = "".join(f"- point {i} about agents [C1] with **bold** and `code` 检索 😀\n" for i in range(400))
        c1 = chunk_ref("C1", "2210.03629v3::s3::001")
        return base_events(rid, "long", [c1]) + tokens("## Long\n\n" + para, 7) + [citation(c1), done(rid)]
    raise KeyError(name)


async def chat_stream(name, rid):
    if name == "split":  # raw bytes cut at arbitrary boundaries, CRLF framing, ping comments, several frames per chunk
        text = f"Unicode: {UNI} — end."
        c1 = chunk_ref("C1", "2210.03629v3::s3::001")
        evs = base_events(rid, "split", [c1]) + [("token", {"delta": ch, "t_ms": 1000 + i}) for i, ch in enumerate(text)]
        evs += [citation(c1), done(rid)]
        raw = "".join((": ping\r\n\r\n" if i % 7 == 0 else "") + frame(e, d).replace("\n", "\r\n") for i, (e, d) in enumerate(evs)).encode()
        i, sizes = 0, [1, 2, 3, 5, 1, 7, 2, 300, 1, 4]
        n = 0
        while i < len(raw):
            s = sizes[n % len(sizes)]
            yield raw[i:i + s]
            i += s
            n += 1
            await asyncio.sleep(0.004)
        return
    if name == "cut":
        for e, d in base_events(rid, "cut", []) + tokens("This partial answer stops here", 4):
            yield frame(e, d).encode()
            await asyncio.sleep(0.02)
        return  # connection ends, no `done`
    if name == "slow":
        for e, d in base_events(rid, "slow", []):
            yield frame(e, d).encode()
        for t in tokens("slow answer " * 40, 6):
            yield frame(*t).encode()
            await asyncio.sleep(0.05)
        yield frame(*done(rid)).encode()
        return
    if name == "stall":
        yield b": ping\n\n"
        await asyncio.sleep(130)
        return
    for e, d in scenario_events(name, rid):
        yield frame(e, d).encode()
        await asyncio.sleep(0.003 if name == "long" else 0.01)


CALLS = []


@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    name = body["query"].split()[0].removeprefix("S:")
    CALLS.append({"name": name, "t": time.time()})
    if name == "http500":
        return PlainTextResponse("<html><body>Internal <b>Server</b> Error<script>window.__xss=20</script></body></html>", 500)
    if name == "http422":
        return JSONResponse({"detail": [{"msg": "query too long <script>window.__xss=21</script>"}, {"msg": "second"}]}, 422)
    rid = f"r{len(CALLS)}{name}"
    return StreamingResponse(chat_stream(name, rid), media_type="text/event-stream")


# ------------------------------------------------------------------------------------------------ browser side
LAST = """(() => { const t = [...document.querySelectorAll('#turns .turn')].pop(); const a = t.querySelector('.answer');
  return {q: t.querySelector('.question').textContent, answer: a.textContent, hidden: a.hidden, streaming: a.classList.contains('streaming'),
          status: t.querySelector('.status').textContent, status_cls: t.querySelector('.status').className,
          badge: t.querySelector('.turn-bar .badge').textContent, links: [...a.querySelectorAll('a.cite')].map(x => x.textContent),
          dropped: [...a.querySelectorAll('.cite-dropped')].map(x => x.textContent), send_disabled: document.getElementById('send').disabled,
          trace_badge: [...document.querySelectorAll('#trace details.trace-turn')].pop().querySelector('summary .badge').textContent}; })()"""

SAFETY = """(() => {
  const all = [...document.querySelectorAll('*')];
  const onattr = all.filter(e => [...e.attributes].some(a => /^on/i.test(a.name))).map(e => e.tagName);
  const bad = document.querySelectorAll('body img, body script:not([src]), body svg, body iframe, body object, body embed');
  const hrefs = [...document.querySelectorAll('a[href]')].map(a => a.getAttribute('href'));
  return {xss: window.__xss || 0, on_attrs: onattr, bad_nodes: [...bad].map(e => e.outerHTML.slice(0, 80)),
          non_http_hrefs: hrefs.filter(h => h !== '#' && !/^https?:\\/\\//i.test(h)),
          inline_scripts: document.querySelectorAll('script:not([src])').length,
          ext_links: [...document.querySelectorAll('.answer a.ext')].map(a => ({h: a.getAttribute('href'), rel: a.rel, target: a.target})),
          literal_evil_in_answer: [...document.querySelectorAll('.answer')].some(a => a.textContent.includes('<script>window.__xss=2</script>')),
          literal_evil_in_trace: document.getElementById('trace').textContent.includes('<img src=x onerror='),
          literal_evil_in_routing: document.getElementById('routing-rows').textContent.includes('<svg onload=')};
})()"""

VIEWER = """({kind: document.getElementById('viewer-kind').textContent, title: document.getElementById('viewer-title').textContent,
  sub: document.getElementById('viewer-sub').textContent, meta: document.getElementById('viewer-meta').textContent,
  text: document.getElementById('viewer-text').textContent,
  link: document.getElementById('viewer-link').getAttribute('href'), link_hidden: document.getElementById('viewer-link').hidden,
  link2: document.getElementById('viewer-link2').getAttribute('href'), link2_hidden: document.getElementById('viewer-link2').hidden})"""

CONTRAST = r"""(() => {
  const parse = (c) => { const m = c.match(/rgba?\(([^)]+)\)/); if (!m) return null; const p = m[1].split(/[ ,/]+/).filter(Boolean).map(Number);
    return {r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1}; };
  const lum = ({r, g, b}) => { const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b); };
  const bgOf = (e) => { const layers = []; for (let n = e; n; n = n.parentElement) { const c = parse(getComputedStyle(n).backgroundColor);
      if (c && c.a > 0) { layers.push(c); if (c.a >= 1) break; } }
    let out = {r: 255, g: 255, b: 255}; for (const l of layers.reverse()) out = {r: l.r * l.a + out.r * (1 - l.a), g: l.g * l.a + out.g * (1 - l.a), b: l.b * l.a + out.b * (1 - l.a)};
    return out; };
  const ratio = (a, b) => { const x = lum(a), y = lum(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); };
  const sels = ['h1', '.banner', '.question', '.answer p', '.answer li', '.answer a.cite', '.answer a.ext', '.answer code', '.answer pre.code',
    '.answer blockquote', '.answer th', '.answer td', '.status', '.status.warn', '.status.error', '.cite-dropped', '.chip.prov-openai',
    '.chip.prov-gemini-openrouter', '.routing-line .label', '.turn-no', '.clock', '.badge.ok', '.badge.warn', '.badge.bad', '.badge.skip',
    '.badge.inj', '.badge.run', '.badge.src-paper', '.badge.src-db', '.badge.src-web', '#trace li', '#trace .t', '#trace .agent-name',
    '#trace .muted', '#routing-rows td', '#routing-rows .muted', '#routing-rows .prov', '#routing-rows .res-bad', '#routing-rows .res-warn',
    '#sources li', '#sources a.cite', '#viewer-title', '#viewer-sub', '#viewer-meta', '#viewer-text', '#viewer-link', 'nav button',
    'textarea', '#send', '.muted', '.flow .node', '.flow .arrow', '#routing-rows tr.turn-head td'];
  const out = {};
  for (const s of sels) { const els = [...document.querySelectorAll(s)].filter(e => e.getClientRects().length || s.startsWith('#viewer') || s.startsWith('#trace') || s.startsWith('#routing') || s.startsWith('#sources'));
    if (!els.length) continue; let min = 99, worst = null;
    for (const e of els.slice(0, 40)) { const st = getComputedStyle(e); const fg = parse(st.color); if (!fg) continue;
      const bg = bgOf(e); const op = Number(st.opacity) || 1; const fgb = {r: fg.r * op + bg.r * (1 - op), g: fg.g * op + bg.g * (1 - op), b: fg.b * op + bg.b * (1 - op)};
      const r = ratio(fgb, bg); if (r < min) { min = r; worst = {fg: st.color, bg: `rgb(${bg.r|0},${bg.g|0},${bg.b|0})`, size: st.fontSize}; } }
    out[s] = {min: Math.round(min * 100) / 100, n: els.length, worst}; }
  return out;
})()"""


async def ask(cdp, q, wait=True, timeout=40):
    await cdp.js(f"document.getElementById('q').value = {json.dumps(q)}; document.getElementById('ask').requestSubmit(); true")
    if not wait:
        return None
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        await asyncio.sleep(0.15)
        s = await cdp.js(LAST)
        if not s["streaming"] and not s["send_disabled"]:
            s["wait_s"] = round(time.perf_counter() - t0, 2)
            return s
    s = await cdp.js(LAST)
    s["timeout"] = True
    return s


async def click_cite(cdp, turn_sel, cid):
    ok = await cdp.js(f"""(() => {{ const a = [...document.querySelectorAll('{turn_sel} .answer a.cite')].find(x => x.textContent === {json.dumps(cid)});
        if (!a) return false; a.click(); return true; }})()""")
    await asyncio.sleep(0.6)
    return ok, await cdp.js(VIEWER)


def check(results, name, cond, detail=None):
    results.append({"id": name, "pass": bool(cond), "detail": detail})


async def run(out, profile, with_stall):
    out.mkdir(parents=True, exist_ok=True)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=APP_PORT, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(50):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{APP_PORT}/").read()
            break
        except Exception:
            time.sleep(0.1)
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={CDP_PORT}", f"--user-data-dir={profile}",
                             "--no-first-run", "--no-default-browser-check", "--window-size=1500,1100", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    R, ev = [], {}
    try:
        for _ in range(60):
            try:
                pages = json.load(urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json/list"))
                page = next(p for p in pages if p["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
            cdp = CDP(ws)
            rd = asyncio.create_task(cdp.reader())
            await cdp.call("Page.enable")
            await cdp.call("Runtime.enable")
            base = f"http://127.0.0.1:{APP_PORT}"
            await cdp.call("Page.navigate", url=base + "/")
            await asyncio.sleep(1.0)
            await cdp.js("window.__xss = 0; true")

            # ---- M1 XSS
            s = await ask(cdp, "S:xss " + EVIL)
            safety = await cdp.js(SAFETY)
            ev["xss_turn"], ev["xss_safety"] = s, safety
            check(R, "M1a no script execution (window.__xss==0)", safety["xss"] == 0, safety["xss"])
            check(R, "M1b no on* attributes / img / svg / iframe / inline script in DOM", not safety["on_attrs"] and not safety["bad_nodes"]
                  and safety["inline_scripts"] == 0, safety)
            check(R, "M1c every <a href> is '#' or http(s)", not safety["non_http_hrefs"], safety["non_http_hrefs"])
            check(R, "M1d hostile text shown literally in answer, trace, routing",
                  safety["literal_evil_in_answer"] and safety["literal_evil_in_trace"] and safety["literal_evil_in_routing"], safety)
            check(R, "M1e external links have rel=noopener noreferrer target=_blank",
                  all("noopener" in l["rel"] and l["target"] == "_blank" for l in safety["ext_links"]) and safety["ext_links"], safety["ext_links"])
            check(R, "M1f unknown markers [C9]/[Z1] are plain text; C1/W1/D1 are links",
                  set(s["links"]) == {"C1", "W1", "D1"} and "C9" in s["answer"], s["links"])
            ok, v = await click_cite(cdp, "#turns .turn:last-child", "C1")
            ev["xss_viewer_C1"] = v
            check(R, "M1g C# viewer: hostile title/text literal, javascript: anchor hidden", ok and EVIL in v["text"] and v["link_hidden"]
                  and v["link2"] == "https://arxiv.org/abs/2210.03629", v)
            ok, v = await click_cite(cdp, "#turns .turn:last-child", "W1")
            ev["xss_viewer_W1"] = v
            check(R, "M1h W# viewer: javascript: URL not linked, snippet literal", ok and v["link_hidden"] and EVIL in v["text"], v)
            ok, v = await click_cite(cdp, "#turns .turn:last-child", "D1")
            ev["xss_viewer_D1"] = v
            check(R, "M1i D# viewer: javascript: href not linked, SQL/rows literal", ok and v["link_hidden"] and "<b>" in v["text"], v)
            safety2 = await cdp.js(SAFETY)
            check(R, "M1j still no execution after viewer clicks", safety2["xss"] == 0 and not safety2["non_http_hrefs"], safety2)
            await shot(cdp, out / "m1_xss.png")

            # ---- M2 byte-split Unicode + CRLF framing
            s = await ask(cdp, "S:split")
            ev["split"] = s
            want = f"Unicode: {UNI} — end."
            check(R, "M2a Unicode across arbitrary byte splits + CRLF + pings reassembled exactly",
                  want in s["answer"] and "�" not in s["answer"] and s["badge"] == "done", s["answer"][:200])
            rtl = await cdp.js("(() => { const t = [...document.querySelectorAll('#turns .turn')].pop(); return {qdir: t.querySelector('.question').dir, adir: t.querySelector('.answer').dir}; })()")
            check(R, "M2b dir=auto on question and answer", rtl == {"qdir": "auto", "adir": "auto"}, rtl)

            # ---- M3 isolation of citation registries across turns
            sA = await ask(cdp, "S:isoA")
            sB = await ask(cdp, "S:isoB")
            okA, vA = await click_cite(cdp, "#turns .turn:nth-last-child(2)", "C1")
            okB, vB = await click_cite(cdp, "#turns .turn:last-child", "C1")
            ev["iso"] = {"A": vA, "B": vB}
            check(R, "M3 same id C1 in two turns opens each turn's own source", okA and okB and "isoA::sec::000" in vA["text"]
                  and "isoB::sec::000" in vB["text"], {"A": vA["meta"], "B": vB["meta"]})

            # ---- M4 encoded chunk id + missing source
            s = await ask(cdp, "S:enc")
            ok, v = await click_cite(cdp, "#turns .turn:last-child", "C1")
            ev["enc"] = v
            check(R, "M4a chunk id with space / # ? & % round-trips through the viewer",
                  ok and "TEXT-OF[2305.10601v2::4 Experiments Game of 24 #1 ?a=b&c%::003]" in v["text"], v["text"][:120])
            ok, v = await click_cite(cdp, "#turns .turn:last-child", "C2")
            check(R, "M4b missing chunk -> visible 'Could not load source: HTTP 404'", ok and "HTTP 404" in v["text"], v)

            # ---- M5 partial / injected
            s = await ask(cdp, "S:partial")
            tr = await cdp.js("(() => { const d = [...document.querySelectorAll('#trace details.trace-turn')].pop(); return {sum: d.querySelector('summary').textContent, agents: [...d.querySelectorAll('details.trace-agent')].map(g => g.querySelector('summary').textContent)}; })()")
            ev["partial"] = {"turn": s, "trace": tr}
            check(R, "M5 partial turn: badge partial, warn status, dropped C2 not linked, injected badges shown",
                  s["badge"] == "partial" and "warn" in s["status_cls"] and s["dropped"] == ["C2"] and "C2" not in s["links"]
                  and "injected" in tr["sum"] and any("Data agent" in a and "failed" in a and "injected" in a for a in tr["agents"]),
                  {"turn": s, "trace": tr})
            await shot(cdp, out / "m5_partial.png")

            # ---- M6 error states
            s = await ask(cdp, "S:errev")
            ev["errev"] = s
            check(R, "M6a non-recoverable error event -> failed badge and error status", s["badge"] == "failed" and "error" in s["status_cls"]
                  and "all providers failed" in s["status"], s)
            s = await ask(cdp, "S:cut")
            ev["cut"] = s
            check(R, "M6b mid-stream close without done -> failed, partial text kept, message says so",
                  s["badge"] == "failed" and "closed before the answer finished" in s["status"] and "partial answer" in s["status"]
                  and "This partial answer" in s["answer"] and not s["send_disabled"], s)
            s = await ask(cdp, "S:http500")
            ev["http500"] = s
            check(R, "M6c HTTP 500 (HTML body) -> failed with HTTP 500, no markup executed", s["badge"] == "failed" and "HTTP 500" in s["status"]
                  and (await cdp.js("window.__xss || 0")) == 0, s)
            s = await ask(cdp, "S:http422")
            ev["http422"] = s
            check(R, "M6d HTTP 422 detail list joined and shown as text", s["badge"] == "failed" and "second" in s["status"]
                  and "<script>" in s["status"], s)
            await shot(cdp, out / "m6_errors.png")

            # ---- M7 double submit
            n0 = len(CALLS)
            await ask(cdp, "S:slow first", wait=False)
            await asyncio.sleep(0.4)
            btn = await cdp.js("document.getElementById('send').disabled")
            await cdp.js("document.getElementById('send').click(); true")  # disabled button click
            await asyncio.sleep(0.2)
            calls_after_click = len(CALLS) - n0
            await cdp.js("""(() => { const q = document.getElementById('q'); q.value = 'S:slow second via Enter';
                q.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', bubbles: true, cancelable: true})); return true; })()""")
            await asyncio.sleep(0.5)
            calls_after_enter = len(CALLS) - n0
            turns_running = await cdp.js("[...document.querySelectorAll('#turns .turn-bar .badge')].filter(b => b.textContent === 'running').length")
            await asyncio.sleep(4.5)
            ev["double_submit"] = {"send_disabled": btn, "calls_after_disabled_click": calls_after_click,
                                   "calls_after_enter": calls_after_enter, "turns_running_concurrently": turns_running}
            check(R, "M7a Ask button disabled while a turn runs (click does nothing)", btn and calls_after_click == 1, ev["double_submit"])
            check(R, "M7b Enter key while a turn runs does not start a second concurrent request", calls_after_enter == 1,
                  ev["double_submit"])
            for _ in range(60):
                if not await cdp.js("document.getElementById('send').disabled"):
                    break
                await asyncio.sleep(0.2)
            await asyncio.sleep(0.5)
            sb = await cdp.js("({disabled: document.getElementById('send').disabled, running: [...document.querySelectorAll('#turns .turn-bar .badge')].filter(b => b.textContent === 'running').length})")
            ev["double_submit"]["after_settle"] = sb
            check(R, "M7c after both settle: no turn left 'running', button enabled", sb == {"disabled": False, "running": 0}, sb)

            # ---- M8 long answer
            await cdp.js("window.__lt = []; new PerformanceObserver(l => l.getEntries().forEach(e => window.__lt.push(e.duration))).observe({entryTypes: ['longtask']}); true")
            t0 = time.perf_counter()
            s = await ask(cdp, "S:long", timeout=90)
            lt = await cdp.js("({n: window.__lt.length, max: Math.max(0, ...window.__lt), sum: window.__lt.reduce((a, b) => a + b, 0)})")
            li = await cdp.js("document.querySelector('#turns .turn:last-child .answer').querySelectorAll('li').length")
            ev["long"] = {"wait_s": s.get("wait_s"), "li": li, "longtasks": lt, "len": len(s["answer"]), "links": len(s["links"])}
            check(R, "M8 long answer (400 list items, ~25k chars) renders completely", li == 400 and len(s["links"]) == 400 and s["badge"] == "done",
                  ev["long"])

            # ---- M9 contrast light and dark (UI now full of every element kind)
            # headless Chrome follows the OS theme, so light must be emulated explicitly too
            await cdp.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "light"}])
            await asyncio.sleep(0.3)
            await cdp.js("document.querySelector('nav button[data-tab=routing]').click(); true")
            light = await cdp.js(CONTRAST)
            light["_scheme_check"] = {"min": 99, "worst": await cdp.js("getComputedStyle(document.body).backgroundColor")}
            await cdp.js("document.querySelector('nav button[data-tab=trace]').click(); true")
            light.update({k: v for k, v in (await cdp.js(CONTRAST)).items() if k.startswith('#trace')})
            await cdp.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "dark"}])
            await asyncio.sleep(0.3)
            dark = await cdp.js(CONTRAST)
            dark["_scheme_check"] = {"min": 99, "worst": await cdp.js("getComputedStyle(document.body).backgroundColor")}
            await shot(cdp, out / "m9_dark_trace.png")
            await cdp.js("document.querySelector('nav button[data-tab=routing]').click(); true")
            dark.update({k: v for k, v in (await cdp.js(CONTRAST)).items() if k.startswith('#routing')})
            await shot(cdp, out / "m9_dark_routing.png")
            await cdp.js("document.querySelectorAll('#sources a.cite')[3].click(); true")
            await asyncio.sleep(0.6)
            dark.update({k: v2 for k, v2 in (await cdp.js(CONTRAST)).items() if k.startswith('#viewer') or k.startswith('#sources')})
            await shot(cdp, out / "m9_dark_viewer.png")
            await cdp.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "light"}])
            ev["contrast"] = {"light": light, "dark": dark}
            low = {m: {k: v for k, v in d.items() if v["min"] < 4.5} for m, d in (("light", light), ("dark", dark))}
            ev["contrast_below_4_5"] = low
            check(R, "M9 text contrast >= 4.5:1 (WCAG AA) for all measured UI text, light and dark", not low["light"] and not low["dark"], low)

            # ---- M10 stall watchdog (60 s)
            if with_stall:
                s = await ask(cdp, "S:stall", timeout=90)
                ev["stall"] = s
                check(R, "M10 silent stream -> watchdog error at ~60 s, turn failed, button re-enabled",
                      s["badge"] == "failed" and "stalled" in s["status"] or "no response" in s["status"], s)

            # ---- M12 a realistic long unbroken token (a pasted URL) in the question must not push the Routing columns off-panel
            url = "https://arxiv.org/html/2305.10601v2?utm_source=newsletter&utm_medium=email&utm_campaign=agentic_ai_reading_group_fall_2026#S3.SS2.p4"
            await ask(cdp, f"S:isoA compare with {url}")
            await cdp.js("document.querySelector('nav button[data-tab=routing]').click(); true")
            await asyncio.sleep(0.3)
            ov = await cdp.js("""(() => { const p = document.getElementById('tab-routing'); const panel = document.getElementById('panel').getBoundingClientRect();
                const th = [...document.querySelectorAll('#tab-routing thead th')].map(h => { const r = h.getBoundingClientRect(); return {t: h.textContent, left: Math.round(r.left), right: Math.round(r.right)}; });
                const tdq = [...document.querySelectorAll('#routing-rows tr.turn-head td')].pop();
                const tt = [...document.querySelectorAll('#trace .turn-title')].pop();
                return {panel_right: Math.round(panel.right), cols: th, scrollW: p.scrollWidth, clientW: p.clientWidth,
                        turn_head_wrap: getComputedStyle(tdq).overflowWrap, trace_title_overflow: tt ? tt.scrollWidth > tt.clientWidth + 1 : null}; })()""")
            await shot(cdp, out / "m12_long_url_routing.png")
            ev["long_url_routing"] = ov
            check(R, "M12 pasted long URL in the question keeps all Routing columns inside the panel",
                  all(c["right"] <= ov["panel_right"] + 1 for c in ov["cols"]), ov)

            errs = [e for e in cdp.events if e.get("method") == "Runtime.exceptionThrown" or e["params"].get("type") in ("error", "warning")]
            ev["console"] = [json.dumps(e)[:300] for e in errs]
            check(R, "M11 no uncaught exceptions / 'bad SSE payload' warnings", not errs, ev["console"][:5])
            rd.cancel()
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except Exception:
            proc.kill()
        server.should_exit = True
    rep = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "mocked": "SSE backend mocked; real static UI and real Chrome",
           "results": R, "passed": sum(r["pass"] for r in R), "failed": sum(not r["pass"] for r in R), "evidence": ev}
    (out / "mock_ui_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    for r in R:
        print(("PASS " if r["pass"] else "FAIL ") + r["id"] + ("" if r["pass"] else "  :: " + json.dumps(r["detail"], ensure_ascii=False, default=str)[:700]))
    print(json.dumps({"passed": rep["passed"], "failed": rep["failed"]}))


if __name__ == "__main__":
    asyncio.run(run(pathlib.Path(sys.argv[1]), sys.argv[2], "--with-stall" in sys.argv))
