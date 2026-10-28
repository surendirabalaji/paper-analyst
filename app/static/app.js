// Paper Analyst chat client: POST + fetch() streaming, incremental SSE parsing, no build step.
// All server text is inserted with textContent / DOM APIs; Markdown is parsed to a data tree (md.js) and
// built with createElement, and links are only made for known citation ids or http(s) URLs.
"use strict";

const $ = (id) => document.getElementById(id);
const MARKERS = /\[([A-Z]\d+(?:\s*[,;]\s*[A-Z]\d+)*)\]/g;
const STALL_MS = 60000; // the server pings every 15 s; this long without a byte means the stream is dead
let turnNo = 0;

// Fault injection for failure-handling demos (Phase 7): #fault=<spec>&fault_token=<token> in the URL. The
// hash never reaches the server or its logs; the server refuses the header unless the token matches. The
// token is kept in memory only and removed from the address bar and history at once (P7-N3).
const FAULT = (() => {
  const h = new URLSearchParams(location.hash.slice(1));
  const f = h.get("fault") && h.get("fault_token") ? { spec: h.get("fault"), token: h.get("fault_token") } : null;
  if (h.has("fault_token")) history.replaceState(null, "", location.pathname + location.search);
  return f;
})();
if (FAULT) {
  const b = el("p", "banner fault",
    `Fault injection is on for this page: ${FAULT.spec}. Failures it causes are test faults, not real outages. ` +
    "The token was removed from the address bar; reload the page to turn fault injection off.");
  document.querySelector("header").append(b);
}
// The injected label once per line: skipped when the message already says so.
const INJ = (d, already = "") => (d && d.injected && !/injected/i.test(already) ? " [injected test fault]" : "");

// ---------------------------------------------------------------- tabs
document.querySelectorAll("nav button").forEach((b) =>
  b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(name) {
  document.querySelectorAll("nav button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  document.querySelectorAll(".tab").forEach((t) => (t.hidden = t.id !== `tab-${name}`));
}

// ---------------------------------------------------------------- SSE parsing
// Frames end with a blank line; a network chunk may hold part of a frame or several frames.
function makeParser(onEvent) {
  let buf = "";
  return (text, flush = false) => {
    buf += text;
    const frames = buf.split(/\r?\n\r?\n/);
    buf = flush ? "" : frames.pop();
    for (const frame of frames) {
      let event = "message";
      const data = [];
      for (const line of frame.split(/\r?\n/)) {
        if (!line || line.startsWith(":")) continue; // keep-alive comment
        const i = line.indexOf(":");
        const field = i < 0 ? line : line.slice(0, i);
        const value = i < 0 ? "" : line.slice(i + 1).replace(/^ /, "");
        if (field === "event") event = value;
        else if (field === "data") data.push(value);
      }
      if (data.length) {
        try { onEvent(event, JSON.parse(data.join("\n"))); }
        catch (e) { console.warn("bad SSE payload", e); }
      }
    }
  };
}

// ---------------------------------------------------------------- rendering helpers
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}

const isHttp = (u) => typeof u === "string" && /^https?:\/\//i.test(u);
const secs = (ms) => (ms == null ? "–" : ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`);

// Plain text with [C1, D2] markers: known ids become citation links, `dropped` ids (attached only to
// statements of missing evidence) are shown struck-through and never linked, unknown ids stay text.
function appendCited(parent, text, ctx) {
  let last = 0;
  for (const m of text.matchAll(MARKERS)) {
    parent.append(document.createTextNode(text.slice(last, m.index)), document.createTextNode("["));
    m[1].split(/\s*[,;]\s*/).forEach((id, i) => {
      if (i) parent.append(document.createTextNode(", "));
      const ref = ctx.registry.get(id);
      if (ref && ctx.dropped.has(id)) {
        const s = el("span", "cite-dropped", id);
        s.title = "Not linked: attached to a statement that the evidence is missing";
        parent.append(s);
      } else if (ref) {
        parent.append(citeLink(id, ref));
      } else {
        parent.append(document.createTextNode(id)); // unknown marker: never a link
      }
    });
    parent.append(document.createTextNode("]"));
    last = m.index + m[0].length;
  }
  parent.append(document.createTextNode(text.slice(last)));
}

function citeLink(id, ref) {
  const a = el("a", `cite cite-${kindOf(ref.type)}`, id);
  a.href = "#";
  a.title = ref.label || id;
  a.addEventListener("click", (e) => { e.preventDefault(); openSource(ref); });
  return a;
}
const kindOf = (t) => (t === "web" ? "web" : isSql(t) ? "db" : "paper");

function buildInline(parent, nodes, ctx) {
  for (const n of nodes) {
    if (n.t === "text") appendCited(parent, n.v, ctx);
    else if (n.t === "code") parent.append(el("code", "", n.v));
    else if (n.t === "link" && isHttp(n.href)) {
      const a = el("a", "ext");
      a.href = n.href;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      buildInline(a, n.c, ctx);
      parent.append(a);
    } else {
      const s = el(n.t === "strong" ? "strong" : n.t === "em" ? "em" : "span");
      buildInline(s, n.c || [], ctx);
      parent.append(s);
    }
  }
}

function buildBlocks(parent, blocks, ctx) {
  for (const b of blocks) {
    if (b.t === "p" || b.t === "h") {
      const n = el(b.t === "p" ? "p" : `h${Math.min(6, b.level + 2)}`); // h1 in an answer renders as h3
      buildInline(n, b.c, ctx);
      parent.append(n);
    } else if (b.t === "hr") parent.append(el("hr"));
    else if (b.t === "code") parent.append(el("pre", "code", b.v));
    else if (b.t === "quote") { const q = el("blockquote"); buildBlocks(q, b.c, ctx); parent.append(q); }
    else if (b.t === "ul" || b.t === "ol") parent.append(buildList(b, ctx));
    else if (b.t === "table") {
      const wrap = el("div", "table-wrap");
      const t = el("table", "md");
      const hr = el("tr");
      b.head.forEach((c) => { const th = el("th"); buildInline(th, c, ctx); hr.append(th); });
      const tb = el("tbody");
      b.rows.forEach((r) => {
        const tr = el("tr");
        r.forEach((c) => { const td = el("td"); buildInline(td, c, ctx); tr.append(td); });
        tb.append(tr);
      });
      const th = el("thead");
      th.append(hr);
      t.append(th, tb);
      wrap.append(t);
      parent.append(wrap);
    }
  }
}

function buildList(b, ctx) {
  const l = el(b.t);
  if (b.t === "ol" && b.start !== 1) l.start = b.start;
  for (const it of b.items) {
    const li = el("li");
    buildInline(li, it.c, ctx);
    if (it.sub) li.append(buildList(it.sub, ctx));
    l.append(li);
  }
  return l;
}

function renderAnswer(node, text, registry, dropped = new Set()) {
  node.replaceChildren();
  buildBlocks(node, MD.parse(text), { registry, dropped });
}

const isSql = (t) => t === "sql_row" || t === "sql_result";
const sqlTable = (cols, rows) => [(cols || []).join(" | "),
  ...(rows || []).map((r) => r.map((v) => (v == null ? "" : String(v))).join(" | "))].join("\n");

// ---------------------------------------------------------------- source viewer
function setViewer({ kind, title, sub, meta, text, link, linkText, link2, link2Text }) {
  showTab("sources");
  $("viewer").hidden = false;
  $("viewer-kind").textContent = kind;
  $("viewer-kind").className = `badge src-${kind}`;
  $("viewer-title").textContent = title;
  $("viewer-sub").textContent = sub || "";
  $("viewer-meta").textContent = meta || "";
  $("viewer-text").textContent = text;
  $("viewer-text").scrollTop = 0;
  for (const [id, href, t] of [["viewer-link", link, linkText], ["viewer-link2", link2, link2Text]]) {
    $(id).hidden = !isHttp(href);
    $(id).href = isHttp(href) ? href : "#";
    $(id).textContent = t || "";
  }
  $("viewer").scrollIntoView({ block: "nearest" });
}

async function openSource(ref) {
  const by = ref.found_by || ref.agent;
  if (isSql(ref.type)) { // database record: the SQL that produced it and the row(s), from the sql_result event
    const rows = ref.rows || [];
    setViewer({
      kind: "database", title: `${ref.cite_id} · ${ref.title || "corpus metadata"}`,
      sub: ref.type === "sql_row" ? "one paper's row in the corpus metadata DB" : `${rows.length} row(s) from the corpus metadata DB`,
      meta: `papers.db (read-only)${by ? ` · found by ${by} via run_sql` : ""}`,
      text: `SQL:\n${ref.sql || "?"}\n\n` +
        (rows.length ? `Result:\n${sqlTable(ref.columns, rows)}` : "Result: 0 rows (no such records in the corpus)"),
      link: ref.source_url, linkText: "Open arXiv abstract page",
    });
    return;
  }
  if (ref.type === "web") { // live web result: the full snippet the model saw, plus the page itself
    const snip = ref.snippet || "";
    setViewer({
      kind: "web", title: `${ref.cite_id} · ${ref.title || ref.source_url}`,
      sub: `${ref.doc_id || ""} · retrieved ${ref.retrieved_at || "?"}`,
      meta: (by ? `found by ${by} via tavily_search (Tavily remote MCP)` : "Tavily remote MCP") +
        (snip ? ` · snippet as given to the model, ${snip.length} characters` : ""),
      text: snip || "(snippet not available; open the page)",
      link: ref.source_url, linkText: "Open web page",
    });
    return;
  }
  setViewer({
    kind: "paper", title: `${ref.cite_id} · ${ref.title || "indexed paper"}`,
    sub: ref.section || "", meta: `chunk ${ref.chunk_id} · loading…`, text: "Loading…",
    link: ref.source_url, linkText: "Open this section on arXiv",
  });
  try {
    const r = await fetch(ref.href);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const s = await r.json();
    setViewer({
      kind: "paper", title: `${ref.cite_id} · ${s.title || ref.title || s.doc_id}`,
      sub: s.section_path || ref.section || "",
      meta: `arXiv:${s.doc_id} · published ${s.published || "?"} · chunk ${s.chunk_id} · ${s.text.length} characters`,
      text: s.text, link: s.html_anchor_url, linkText: "Open this section on arXiv",
      link2: s.abs_url, link2Text: "Abstract page",
    });
  } catch (e) {
    $("viewer-text").textContent = `Could not load source: ${e.message}`;
    $("viewer-meta").textContent = `chunk ${ref.chunk_id}`;
  }
}

// ---------------------------------------------------------------- trace (grouped per turn and per agent)
const AGENT_LABEL = { orchestration: "Orchestration (planner + supervisor)", research: "Research agent",
  summary: "Summary agent", data: "Data agent", synthesis: "Synthesis" };
const NODE_LABEL = { start: "Request", planner: "Planner", supervisor: "Supervisor", research: "Research",
  summary: "Summary", data: "Data", synthesize: "Synthesis" };
const groupOf = (agent) => (!agent || agent === "planner" || agent === "supervisor" ? "orchestration"
  : agent === "synthesizer" || agent === "synthesize" ? "synthesis" : agent);
const RANK = { running: 0, done: 1, skipped: 2, partial: 3, failed: 4 };
const badgeClass = (s) => ({ done: "ok", ok: "ok", partial: "warn", skipped: "skip", failed: "bad", error: "bad",
  running: "run", injected: "inj" }[s] || "");
const statusKind = (s) => (s === "done" || s === "ok" ? "ok" : s === "partial" || s === "skipped" ? "warn" : "bad");

function makeTrace(turn, query) {
  const det = el("details", "trace-turn");
  det.open = true;
  [...$("trace").querySelectorAll("details.trace-turn")].forEach((d) => (d.open = false)); // older turns fold
  const sum = el("summary");
  const badge = el("span", "badge run", "running");
  sum.append(badge, el("span", "turn-title", `Turn #${turn}: ${query}`));
  const flow = el("div", "flow");
  flow.append(el("span", "node", "Request"));
  det.append(sum, flow);
  $("trace").append(det);
  const groups = new Map();
  const T = {
    det, injected: false,
    group(key) {
      if (groups.has(key)) return groups.get(key);
      const g = el("details", `trace-agent agent-${key}`);
      g.open = true;
      const s = el("summary");
      const b = el("span", "badge run", "running");
      const extra = el("span", "badges");
      const count = el("span", "muted count", "");
      s.append(el("span", "agent-name", AGENT_LABEL[key] || `${key} agent`), b, extra, count);
      const ol = el("ol");
      g.append(s, ol);
      det.append(g);
      const rec = { g, b, extra, count, ol, status: "running", n: 0, tasks: new Set() };
      groups.set(key, rec);
      return rec;
    },
    add(key, label, text, kind = "", t_ms) {
      const g = T.group(key);
      const li = el("li", kind ? `row-${kind}` : "");
      li.append(el("span", "t", t_ms != null ? `+${secs(t_ms)}` : ""), el("span", `badge ${kind}`, label),
        document.createTextNode(` ${text}`));
      g.ol.append(li);
      g.count.textContent = `${++g.n} event${g.n > 1 ? "s" : ""}`;
      li.scrollIntoView({ block: "nearest" });
    },
    status(key, status, injected) { // worst status wins when an agent ran several tasks
      const g = T.group(key);
      if (g.status === "running" || RANK[status] > RANK[g.status]) g.status = status;
      g.b.textContent = g.status;
      g.b.className = `badge ${badgeClass(g.status)}`;
      if (injected && !g.extra.querySelector(".inj")) g.extra.append(el("span", "badge inj", "injected"));
      if (injected && !T.injected) { T.injected = true; sum.append(el("span", "badge inj", "injected")); }
    },
    flow(name, cls = "") {
      flow.append(el("span", "arrow", "→"), el("span", `node ${cls}`, name));
    },
    finish(status) {
      badge.textContent = status;
      badge.className = `badge ${badgeClass(status)}`;
      for (const g of groups.values()) if (g.status === "running") { g.b.textContent = "ended"; g.b.className = "badge"; }
    },
  };
  return T;
}

// Supervisor/graph transitions as sentences.
function describeTransition(d) {
  const to = NODE_LABEL[d.to] || d.to;
  const from = NODE_LABEL[d.from] || d.from;
  if (d.from === "start") return `Request received; the ${to.toLowerCase()} decides which workers to use.`;
  if (d.from === "supervisor" && d.task_id) return `Supervisor hands task ${d.task_id} to ${to} (${d.reason}).`;
  if (d.from === "supervisor" && d.to === "synthesize") return `Supervisor sends the evidence to Synthesis (${d.reason}).`;
  if (d.to === "supervisor") return `${from} returns task ${d.task_id || ""} to the Supervisor: ${d.reason.replace(/^\S+\s/, "")}.`;
  return `${from} → ${to}: ${d.reason}`;
}

// ---------------------------------------------------------------- routing
const SHORT = { openai: "OpenAI", "gemini-openrouter": "Gemini" };
const fmtCost = (c, src) => (c == null ? "" : ` · $${c < 0.001 ? c.toFixed(5) : c.toFixed(4)}` +
  (src === "estimated" ? " (est.)" : src === "reported" ? " (reported)" : ""));
const fmtInputs = (f) => Object.entries(f || {})
  .map(([k, v]) => `${k}=${Array.isArray(v) ? v.join("/") : v}`).join(", ");

// One Routing-tab row per decision (original or fallback); the result cell is filled by llm_result.
function routeRow(turn, d, kind) {
  const tr = el("tr", kind);
  const why = el("td");
  why.append(el("strong", "", `${d.dimension}: `), document.createTextNode(d.rationale));
  if (d.features && Object.keys(d.features).length) why.append(el("div", "muted", `inputs: ${fmtInputs(d.features)}`));
  const prov = el("td");
  prov.append(el("span", `prov prov-${d.provider}`, d.provider_label), document.createTextNode(` · ${d.model_id}`));
  if (d.reasoning_effort) prov.append(el("div", "muted", `reasoning effort: ${d.reasoning_effort}`));
  tr.append(el("td", "", `#${turn}`), el("td", "", `${d.agent} / ${d.call_site}`), prov, why, el("td", "", "…"));
  $("routing-rows").append(tr);
  return tr;
}

// ---------------------------------------------------------------- one turn
async function ask(query) {
  const turn = ++turnNo;
  const t0 = performance.now();
  const box = el("div", "turn");
  const head = el("div", "turn-head");
  const tBadge = el("span", "badge run", "running");
  const clock = el("span", "muted clock", "0.0 s");
  head.append(el("span", "turn-no", `#${turn}`), tBadge, clock);
  const question = el("div", "question", query);
  question.dir = "auto";
  const routeLine = el("div", "routing-line");
  const answer = el("div", "answer streaming");
  answer.dir = "auto";
  const status = el("div", "status", "Planning…");
  box.append(head, question, routeLine, answer, status);
  $("turns").append(box);
  box.scrollIntoView({ block: "end" });
  const tick = setInterval(() => (clock.textContent = secs(Math.round(performance.now() - t0))), 200);

  const registry = new Map();
  const dropped = new Set();
  const routes = new Map();
  const picks = []; // per-turn routing summary: [{agent, provider, dimension, fallback}]
  const T = makeTrace(turn, query);
  const paintRouting = () => {
    const groups = [];
    for (const p of picks) {
      const last = groups[groups.length - 1];
      const key = `${p.agent}|${p.provider}|${p.dimension}|${p.fallback}`;
      if (last && last.key === key) last.n++;
      else groups.push({ key, ...p, n: 1 });
    }
    routeLine.replaceChildren(el("span", "label", "Routing: "));
    groups.forEach((g, i) => {
      if (i) routeLine.append(document.createTextNode(" · "));
      const chip = el("span", `chip prov-${g.provider}${g.fallback ? " fb" : ""}`,
        `${g.agent} → ${SHORT[g.provider] || g.provider}${g.n > 1 ? ` ×${g.n}` : ""} (${g.dimension})` +
        (g.fallback ? " ↪ fallback" : ""));
      chip.title = g.model || "";
      routeLine.append(chip);
    });
  };
  const rh = el("tr", "turn-head");
  const rth = el("td", "", `Turn #${turn}: ${query}`);
  rth.colSpan = 5;
  rh.append(rth);
  $("routing-rows").append(rh);
  let text = "";
  let gotDone = false;
  let firstToken = false;
  let failed = false;
  let pending = false;
  const repaint = () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; renderAnswer(answer, text, registry, dropped); });
  };
  const fail = (msg) => { // a terminal error: the status line settles on it, never on "Planning…"
    failed = true;
    status.textContent = `Error: ${msg}`;
    status.classList.add("error");
  };

  const handlers = {
    meta: (d) => {
      T.add("orchestration", "start", `request ${d.request_id} · k=${d.k} · workers available: ${d.workers.join(", ")}` +
        ` · budget ${d.budget_s} s · single-turn`, "", d.t_ms);
      if (d.fault_injection) T.add("orchestration", "fault injection",
        `${d.fault_injection.rules.join(", ")} (${d.fault_injection.source}): ${d.fault_injection.note}`, "inj", d.t_ms);
    },
    fault_injected: (d) => T.add(groupOf(d.agent || (d.kind === "worker" ? d.name : "")), "FAULT",
      `injected ${d.target}` + (d.agent ? ` in ${d.agent}` : "") + (d.call_site ? ` / ${d.call_site}` : "") +
      (d.task_id ? ` (${d.task_id})` : "") + ` — ${d.note}`, "inj", d.t_ms),
    plan: (d) => {
      T.add("orchestration", "plan", `${d.plan_summary} (${d.source === "fallback" ? "fallback plan" : "planner"})`,
        d.source === "fallback" ? "warn" : "", d.t_ms);
      d.tasks.forEach((t) => T.add("orchestration", `task ${t.id}`, `${t.worker}${t.needs_live ? " [needs live web]" : ""}: ` +
        `${t.instruction}` + (t.depends_on.length ? ` (after ${t.depends_on.join(", ")})` : ""), "", d.t_ms));
      d.notes.forEach((n) => T.add("orchestration", "note", n, "warn", d.t_ms));
      status.textContent = d.tasks.length ? `Plan: ${d.tasks.map((t) => `${t.id} ${t.worker}`).join(" → ")}` : "No research needed";
      T.status("orchestration", d.source === "fallback" ? "partial" : "done");
    },
    transition: (d) => {
      T.add("orchestration", "→", describeTransition(d), "", d.t_ms);
      if (d.from === "supervisor" && d.to !== "supervisor") {
        T.flow(`${NODE_LABEL[d.to] || d.to}${d.task_id ? ` ${d.task_id}` : ""}`, d.to === "synthesize" ? "synth" : "");
        if (!firstToken) status.textContent = d.to === "synthesize" ? "Synthesis: writing the answer…"
          : `${NODE_LABEL[d.to] || d.to} agent working on ${d.task_id}…`;
      } else if (d.from === "start") T.flow("Planner");
    },
    llm_route: (d) => {
      routes.set(d.decision_id, routeRow(turn, d, ""));
      picks.push({ agent: d.agent, provider: d.provider, dimension: d.dimension, fallback: false, model: d.model_id });
      paintRouting();
      T.add(groupOf(d.agent), "route", `${d.call_site} → ${d.provider_label} ${d.model_id} (${d.dimension})`, "route", d.t_ms);
    },
    llm_retry: (d) => T.add(groupOf(d.agent), "retry", `${d.call_site} on ${d.provider}: ${d.error_kind}${INJ(d)}, ` +
      `retrying in ${d.backoff_s}s`, "warn", d.t_ms),
    fallback: (d) => {
      routes.set(d.decision_id, routeRow(turn, d, "fallback"));
      picks.push({ agent: d.agent, provider: d.provider, dimension: "fallback", fallback: true, model: d.model_id });
      paintRouting();
      T.add(groupOf(d.agent), "fallback", `${d.call_site}: ${d.from_provider} failed (${d.error_kind})${INJ(d)} → ` +
        `${d.to_provider}`, "warn", d.t_ms);
    },
    llm_result: (d) => {
      const tr = routes.get(d.decision_id);
      if (!tr) return;
      const cell = tr.lastChild;
      const line = el("div", d.status === "ok" ? "" : `res-${statusKind(d.status)}`,
        (d.attempt > 1 ? `try ${d.attempt}: ` : "") +
        (d.status === "ok"
          ? `${d.input_tokens}→${d.output_tokens} tok` + (d.reasoning_tokens ? ` (${d.reasoning_tokens} reasoning)` : "") +
            ` · ${d.latency_ms} ms` + (d.ttft_ms != null ? ` · TTFT ${d.ttft_ms} ms` : "") + fmtCost(d.cost_usd, d.cost_source)
          : `${d.status}${d.error_kind ? ` (${d.error_kind})` : ""}${INJ(d)} after ${d.latency_ms} ms`));
      if (d.upstream_provider) line.append(el("div", "muted", `served by ${d.upstream_provider}`));
      if (cell.textContent === "…") cell.replaceChildren();
      cell.append(line);
    },
    tool_call: (d) => T.add(groupOf(d.agent), "tool", `calls ${d.tool}(${JSON.stringify(d.args_safe)})`, "", d.t_ms),
    tool_result: (d) => T.add(groupOf(d.agent), d.status === "ok" ? "result" : d.status, `${d.tool} → ${d.status}` +
      (d.error_kind ? ` (${d.error_kind}, recoverable)` : "") + INJ(d) +
      (d.n_results != null ? `, ${d.n_results} results` : "") + `, ${d.latency_ms} ms`, d.injected ? "inj" : statusKind(d.status), d.t_ms),
    tool_gate: (d) => T.add(groupOf(d.agent), "gate", `${d.tool} ${d.enabled ? "enabled" : "not enabled"}: ${d.reason}`,
      d.enabled ? "mcp" : "", d.t_ms),
    web_search: (d) => {
      d.results.forEach((r) => registry.set(r.cite_id, {
        ...r, type: "web", doc_id: r.domain, source_url: r.url, href: r.url, agent: d.agent,
        label: `${r.title} — web, retrieved ${r.retrieved_at}`,
      }));
      T.add(groupOf(d.agent), `${d.agent} ⇄ MCP`, `${d.tool} on ${d.server} (${d.purpose}) "${d.query}": ` +
        (d.status === "ok"
          ? d.results.map((r) => `${r.cite_id} ${r.domain}`).join(", ") || "no results"
          : `${d.error_kind}, continuing without web evidence (${d.message_safe})`) + `, ${d.latency_ms} ms`,
        d.status === "ok" ? "mcp" : "warn", d.t_ms);
    },
    retrieval: (d) => {
      d.chunks.forEach((c) => registry.set(c.cite_id, { ...c, label: `${c.title} — ${c.section}` }));
      T.add(groupOf(d.agent), "retrieval", `${d.tool ? `${d.tool} ` : ""}"${d.query}" → ` +
        `${d.chunks.map((c) => `${c.cite_id} ${c.doc_id} (${c.score.toFixed(2)})`).join(", ") || "no chunks"} in ${d.retrieval_ms} ms`,
        "", d.t_ms);
    },
    sql_result: (d) => {
      (d.refs || []).forEach((r) => registry.set(r.cite_id, {
        ...r, sql: d.sql, columns: d.columns, source_url: r.href, agent: d.agent,
        label: `${r.title} — corpus metadata (SQL)`,
      }));
      T.add(groupOf(d.agent), `${d.agent} ⇄ SQLite`, d.status === "ok"
        ? `run_sql (read-only) ${d.sql} → ${d.n_rows} row(s)${d.truncated ? " (capped at 50)" : ""} in ${d.elapsed_ms} ms: ` +
          (d.cite_ids.length ? d.cite_ids.join(" ") : "none")
        : `run_sql ${d.error_kind}: ${d.message_safe} · SQL: ${d.sql}` +
          (d.repair_rounds_left ? " · one repair round left" : " · no repair rounds left"),
        d.status === "ok" ? "mcp" : "warn", d.t_ms);
    },
    handoff: (d) => T.add(groupOf(d.to_agent), "handoff", `${d.from_tasks.join(", ")} → ${d.to_task} ${d.to_agent}: ` +
      (d.papers.length ? d.papers.map((p) => `arXiv:${p.arxiv_id}${p.cite_id ? ` [${p.cite_id}]` : ""}`).join(", ")
        : "no paper ids") + (d.evidence_ids.length ? ` · upstream evidence ${d.evidence_ids.join(" ")}` : ""), "", d.t_ms),
    worker_result: (d) => {
      const why = d.reason ? `: ${d.reason}` : "";
      T.add(groupOf(d.agent), d.status, `${d.task_id} ${d.status}` + (d.attempt > 1 ? ` (attempt ${d.attempt})` : "") +
        (d.error_kind ? ` (${d.error_kind})` : "") + why + INJ(d, why) +
        (d.evidence_ids && d.evidence_ids.length ? `, cites ${d.evidence_ids.join(" ")}` : ""),
        d.injected ? "inj" : statusKind(d.status), d.t_ms);
      T.status(groupOf(d.agent), d.status, d.injected);
    },
    token: (d) => {
      if (!firstToken) {
        firstToken = true;
        status.textContent = `Streaming answer… (first token after ${secs(d.t_ms)})`;
        T.add("synthesis", "first token", `the answer starts streaming`, "ok", d.t_ms);
      }
      text += d.delta;
      repaint();
    },
    citation: (d) => {
      const ref = { ...(registry.get(d.cite_id) || {}), ...d }; // keeps fields from retrieval / web_search
      registry.set(d.cite_id, ref);
      const li = el("li");
      li.append(el("span", "badge", `#${turn}`), el("span", `badge src-${kindOf(d.type)}`, kindOf(d.type)),
        citeLink(d.cite_id, ref), document.createTextNode(
          d.type === "web" ? ` ${d.title || d.label} (${d.doc_id})` : isSql(d.type) ? ` ${d.label}`
            : ` ${d.title || d.label}${d.section ? ` — ${d.section}` : ""} (arXiv:${d.doc_id})`));
      $("sources").append(li);
      repaint();
    },
    error: (d) => {
      const who = d.agent || (d.scope === "synthesize" ? "synthesizer" : d.scope === "planner" ? "planner" : "");
      T.add(groupOf(who), "error", `${d.scope}${d.agent ? " " + d.agent : ""}${d.error_kind ? ` (${d.error_kind})` : ""}: ` +
        `${d.message_safe}${INJ(d, d.message_safe)}${d.recoverable ? " · recoverable, the request continues" : ""}`,
        d.injected ? "inj" : d.recoverable ? "warn" : "bad", d.t_ms);
      if (d.scope === "synthesize") T.status("synthesis", "failed", d.injected);
      if (d.scope === "planner") T.status("orchestration", "partial", d.injected);
      if (!d.recoverable) fail(d.message_safe);
    },
    done: (d) => {
      gotDone = true;
      const t = d.timings;
      const tok = Object.entries(d.tokens_by_provider).map(([p, v]) => `${SHORT[p] || p} ${v.calls} calls, ${v.input_tokens}→${v.output_tokens} tok`).join("; ") +
        (d.cost_usd != null ? ` · cost $${d.cost_usd.toFixed(4)}` +
          (d.cost_basis && d.cost_basis.estimated ? ` (${d.cost_basis.estimated} call(s) estimated)` : "") : "") + (d.fallbacks ? ` · ${d.fallbacks} fallback(s)` : "");
      if (!failed) {
        status.textContent = `${d.status} · TTFT ${secs(t.ttft_ms)} · total ${secs(t.total_ms)} · retrieval ${t.retrieval_ms} ms · ${tok}` +
          (d.unresolved_citations.length ? ` · unresolved: ${d.unresolved_citations.join(" ")}` : "") +
          (d.dropped_citations && d.dropped_citations.length ? ` · not linked (missing-evidence statements): ${d.dropped_citations.join(" ")}` : "");
        if (d.status !== "done") status.classList.add("warn");
      }
      d.dropped_citations && d.dropped_citations.forEach((c) => dropped.add(c));
      repaint();
      if (T.det.querySelector(".agent-synthesis") || firstToken) T.status("synthesis", d.status === "done" ? "done" : "partial");
      T.flow(d.status === "done" ? "Done" : d.status, d.status === "done" ? "ok" : "warn");
      T.add("orchestration", "done", `${d.status} in ${secs(t.total_ms)}` +
        (d.tasks && d.tasks.length ? " · " + d.tasks.map((x) => `${x.id} ${x.worker} ${x.status}`).join(", ") : "") +
        (d.faults_injected && d.faults_injected.length ? ` · ${d.faults_injected.length} injected fault(s)` : ""),
        statusKind(d.status), d.t_ms);
      T.finish(failed ? "failed" : d.status);
    },
  };

  const parse = makeParser((ev, data) => (handlers[ev] || (() => {}))(data));
  const ctrl = new AbortController();
  let stalled = false;
  let stall = setTimeout(() => { stalled = true; ctrl.abort(); }, STALL_MS);
  const alive = () => { clearTimeout(stall); stall = setTimeout(() => { stalled = true; ctrl.abort(); }, STALL_MS); };
  try {
    let r;
    try {
      r = await fetch("/api/chat", {
        method: "POST",
        signal: ctrl.signal,
        headers: { "Content-Type": "application/json", Accept: "text/event-stream",
          ...(FAULT ? { "X-Fault-Inject": FAULT.spec, "X-Fault-Token": FAULT.token } : {}) },
        body: JSON.stringify({ query }),
      });
    } catch (e) {
      throw new Error(stalled ? `no response from the server for ${STALL_MS / 1000} s`
        : `could not reach the server (${e.message})`);
    }
    if (!r.ok) {
      const body = await r.text();
      let detail = body.slice(0, 300);
      try {
        const j = JSON.parse(body);
        detail = typeof j.detail === "string" ? j.detail
          : Array.isArray(j.detail) ? j.detail.map((x) => x.msg).join("; ") : detail;
      } catch { /* not JSON: show the text */ }
      throw new Error(`the server refused the request (HTTP ${r.status}): ${detail || r.statusText}`);
    }
    const reader = r.body.getReader();
    const dec = new TextDecoder("utf-8"); // stream mode keeps multi-byte characters split across chunks intact
    for (;;) {
      let chunk;
      try { chunk = await reader.read(); }
      catch (e) {
        throw new Error(stalled ? `the stream stalled (no data for ${STALL_MS / 1000} s) and was stopped`
          : `the connection was lost (${e.message})`);
      }
      alive();
      if (chunk.done) { parse(dec.decode(), true); break; }
      parse(dec.decode(chunk.value, { stream: true }));
    }
    if (!gotDone) throw new Error("the connection closed before the answer finished" + (text ? "; the partial answer is shown above" : ""));
  } catch (e) {
    fail(e.message);
    T.add("orchestration", "error", e.message, "bad", Math.round(performance.now() - t0));
    T.finish("failed");
  } finally {
    clearTimeout(stall);
    clearInterval(tick);
    clock.textContent = secs(Math.round(performance.now() - t0));
    answer.classList.remove("streaming");
    renderAnswer(answer, text, registry, dropped);
    if (!text) answer.hidden = true;
    const final = failed ? "failed" : T.det.querySelector("summary .badge").textContent;
    tBadge.textContent = final;
    tBadge.className = `badge ${badgeClass(final)}`;
  }
}

$("ask").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("q").value.trim();
  if (!q) return;
  $("send").disabled = true;
  $("q").value = "";
  try { await ask(q); } finally { $("send").disabled = false; $("q").focus(); }
});
$("q").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("ask").requestSubmit(); }
});
