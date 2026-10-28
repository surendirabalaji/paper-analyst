// Paper Analyst chat client: POST + fetch() streaming, incremental SSE parsing, no build step.
// All server text is inserted with textContent; links are built with DOM APIs only for known citation ids.
"use strict";

const $ = (id) => document.getElementById(id);
const MARKERS = /\[([A-Z]\d+(?:\s*[,;]\s*[A-Z]\d+)*)\]/g;
let turnNo = 0;

// Fault injection for failure-handling demos (Phase 7): #fault=<spec>&fault_token=<token> in the URL. The
// hash never reaches the server or its logs; the server refuses the header unless the token matches.
const FAULT = (() => {
  const h = new URLSearchParams(location.hash.slice(1));
  return h.get("fault") && h.get("fault_token") ? { spec: h.get("fault"), token: h.get("fault_token") } : null;
})();
if (FAULT) {
  const b = document.createElement("p");
  b.className = "banner fault";
  b.textContent = `Fault injection is on for your requests: ${FAULT.spec}. Failures it causes are test faults, not real outages.`;
  document.querySelector("header").append(b);
}
const INJ = (d) => (d && d.injected ? " [injected test fault]" : "");

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

// `dropped`: markers attached only to statements of missing evidence; shown as plain text, never linked.
function renderAnswer(node, text, registry, dropped = new Set()) {
  node.replaceChildren();
  let last = 0;
  for (const m of text.matchAll(MARKERS)) {
    node.append(document.createTextNode(text.slice(last, m.index)));
    node.append(document.createTextNode("["));
    m[1].split(/\s*[,;]\s*/).forEach((id, i) => {
      if (i) node.append(document.createTextNode(", "));
      const ref = registry.get(id);
      if (ref && dropped.has(id)) {
        const s = el("span", "cite-dropped", id);
        s.title = "Not linked: attached to a statement that the evidence is missing";
        node.append(s);
      } else if (ref) {
        const a = el("a", "cite", id);
        a.href = "#";
        a.title = ref.label;
        a.addEventListener("click", (e) => { e.preventDefault(); openSource(ref); });
        node.append(a);
      } else {
        node.append(document.createTextNode(id)); // unknown marker: never a link
      }
    });
    node.append(document.createTextNode("]"));
    last = m.index + m[0].length;
  }
  node.append(document.createTextNode(text.slice(last)));
}

const isSql = (t) => t === "sql_row" || t === "sql_result";
const sqlTable = (cols, rows) => [(cols || []).join(" | "),
  ...(rows || []).map((r) => r.map((v) => (v == null ? "" : String(v))).join(" | "))].join("\n");

async function openSource(ref) {
  showTab("sources");
  const v = $("viewer");
  v.hidden = false;
  $("viewer-link").hidden = false;
  if (isSql(ref.type)) { // database record: the SQL that produced it and the row(s), from the sql_result event
    const rows = ref.rows || [];
    $("viewer-title").textContent = `${ref.cite_id} · ${ref.title}`;
    $("viewer-meta").textContent = `corpus metadata DB (papers.db, read-only) · ${ref.type === "sql_row" ? "one paper's row" :
      `${rows.length} row(s)`} · found by ${ref.found_by || ref.agent || "data"} via run_sql`;
    $("viewer-text").textContent = `SQL:\n${ref.sql || "?"}\n\n` +
      (rows.length ? `Result:\n${sqlTable(ref.columns, rows)}` : "Result: 0 rows (no such records in the corpus)");
    $("viewer-link").textContent = "Open arXiv abstract page";
    $("viewer-link").href = isHttp(ref.source_url) ? ref.source_url : "#";
    $("viewer-link").hidden = !isHttp(ref.source_url);
    return;
  }
  if (ref.type === "web") { // live web result: snippet as retrieved by Tavily, plus the page itself
    $("viewer-title").textContent = `${ref.cite_id} · ${ref.title || ref.source_url}`;
    $("viewer-meta").textContent = `web · ${ref.doc_id} · retrieved ${ref.retrieved_at || "?"}` +
      (ref.found_by || ref.agent ? ` · found by ${ref.found_by || ref.agent} via tavily_search` : "");
    $("viewer-text").textContent = ref.snippet || "(snippet not available; open the page)";
    $("viewer-link").textContent = "Open web page";
    $("viewer-link").href = isHttp(ref.source_url) ? ref.source_url : "#";
    return;
  }
  $("viewer-link").textContent = "Open on arXiv";
  $("viewer-title").textContent = `${ref.cite_id} · ${ref.title || ref.chunk_id}`;
  $("viewer-meta").textContent = `${ref.section || ""} · chunk ${ref.chunk_id}`;
  $("viewer-text").textContent = "Loading…";
  $("viewer-link").href = ref.source_url || "#";
  try {
    const r = await fetch(ref.href);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const s = await r.json();
    $("viewer-text").textContent = s.text;
    $("viewer-meta").textContent = `arXiv:${s.doc_id} · ${s.section_path} · published ${s.published} · chunk ${s.chunk_id}`;
    $("viewer-link").href = s.html_anchor_url;
  } catch (e) {
    $("viewer-text").textContent = `Could not load source: ${e.message}`;
  }
}

function traceItem(turn, badge, text, kind = "") {
  const li = el("li");
  li.append(el("span", `badge ${kind}`, `#${turn} ${badge}`), document.createTextNode(text));
  $("trace").append(li);
  li.scrollIntoView({ block: "nearest" });
}

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
  const prov = el("td", "", `${d.provider_label} · ${d.model_id}`);
  if (d.reasoning_effort) prov.append(el("div", "muted", `reasoning effort: ${d.reasoning_effort}`));
  tr.append(el("td", "", `#${turn}`), el("td", "", `${d.agent} / ${d.call_site}`), prov, why, el("td", "", "…"));
  $("routing-rows").append(tr);
  return tr;
}

const statusKind = (s) => (s === "done" || s === "ok" ? "ok" : s === "partial" || s === "skipped" ? "warn" : "bad");

// ---------------------------------------------------------------- one turn
async function ask(query) {
  const turn = ++turnNo;
  const box = el("div", "turn");
  const answer = el("div", "answer streaming");
  const status = el("div", "status", "Planning…");
  box.append(el("div", "question", query), answer, status);
  $("turns").append(box);
  box.scrollIntoView({ block: "end" });

  const registry = new Map();
  const dropped = new Set();
  const routes = new Map();
  const picks = []; // per-turn routing summary: [{agent, provider, dimension, fallback}]
  const routeLine = el("div", "routing-line");
  box.append(routeLine);
  const paintRouting = () => {
    const groups = [];
    for (const p of picks) {
      const last = groups[groups.length - 1];
      const key = `${p.agent}|${p.provider}|${p.dimension}|${p.fallback}`;
      if (last && last.key === key) last.n++;
      else groups.push({ key, ...p, n: 1 });
    }
    routeLine.textContent = "Routing: " + groups.map((g) =>
      `${g.agent} → ${SHORT[g.provider] || g.provider}${g.n > 1 ? ` ×${g.n}` : ""} (${g.dimension})` +
      (g.fallback ? " ↪ fallback" : "")).join(" · ");
  };
  const head = el("tr", "turn-head");
  const th = el("td", "", `Turn #${turn}: ${query}`);
  th.colSpan = 5;
  head.append(th);
  $("routing-rows").append(head);
  let text = "";
  let gotDone = false;
  let pending = false;
  const repaint = () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; renderAnswer(answer, text, registry, dropped); });
  };

  const handlers = {
    meta: (d) => {
      traceItem(turn, "start", ` request ${d.request_id}, k=${d.k}, workers: ${d.workers.join(", ")}`);
      if (d.fault_injection) traceItem(turn, "fault injection", ` ${d.fault_injection.rules.join(", ")} (${d.fault_injection.source}): ${d.fault_injection.note}`, "bad");
    },
    fault_injected: (d) => traceItem(turn, "FAULT", ` injected ${d.target}` +
      (d.agent ? ` in ${d.agent}` : "") + (d.call_site ? ` / ${d.call_site}` : "") + (d.task_id ? ` (${d.task_id})` : "") +
      ` — ${d.note}`, "bad"),
    plan: (d) => {
      traceItem(turn, "plan", ` ${d.plan_summary} (${d.source})`);
      d.tasks.forEach((t) => traceItem(turn, t.id, ` ${t.worker}${t.needs_live ? " [needs_live]" : ""}: ${t.instruction}` +
        (t.depends_on.length ? ` (after ${t.depends_on.join(", ")})` : "")));
      d.notes.forEach((n) => traceItem(turn, "note", ` ${n}`, "warn"));
      status.textContent = d.tasks.length ? `Plan: ${d.tasks.map((t) => t.worker).join(" → ")}` : "No research needed";
    },
    transition: (d) => {
      traceItem(turn, "→", ` ${d.from} → ${d.to}: ${d.reason}`);
      if (d.to !== "supervisor") status.textContent = `${d.to}…`;
    },
    llm_route: (d) => {
      routes.set(d.decision_id, routeRow(turn, d, ""));
      picks.push({ agent: d.agent, provider: d.provider, dimension: d.dimension, fallback: false });
      paintRouting();
    },
    llm_retry: (d) => {
      traceItem(turn, "retry", ` ${d.agent} ${d.call_site} on ${d.provider}: ${d.error_kind}${INJ(d)}, retrying in ${d.backoff_s}s`, "warn");
    },
    fallback: (d) => {
      routes.set(d.decision_id, routeRow(turn, d, "fallback"));
      picks.push({ agent: d.agent, provider: d.provider, dimension: "fallback", fallback: true });
      paintRouting();
      traceItem(turn, "fallback", ` ${d.agent} ${d.call_site}: ${d.from_provider} failed (${d.error_kind})${INJ(d)} → ${d.to_provider}`, "warn");
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
    tool_call: (d) => traceItem(turn, d.agent, ` calls ${d.tool}(${JSON.stringify(d.args_safe)})`),
    tool_result: (d) => traceItem(turn, d.agent, ` ${d.tool} → ${d.status}` +
      (d.error_kind ? ` (${d.error_kind}, recoverable)` : "") + INJ(d) +
      (d.n_results != null ? `, ${d.n_results} results` : "") + `, ${d.latency_ms} ms`, statusKind(d.status)),
    tool_gate: (d) => traceItem(turn, d.agent, ` ${d.tool} ${d.enabled ? "enabled" : "not enabled"}: ${d.reason}`,
      d.enabled ? "mcp" : ""),
    web_search: (d) => {
      d.results.forEach((r) => registry.set(r.cite_id, {
        ...r, type: "web", doc_id: r.domain, source_url: r.url, href: r.url, agent: d.agent,
        label: `${r.title} — web, retrieved ${r.retrieved_at}`,
      }));
      traceItem(turn, `${d.agent} ⇄ MCP`, ` ${d.tool} on ${d.server} (${d.purpose}) "${d.query}": ` +
        (d.status === "ok"
          ? d.results.map((r) => `${r.cite_id} ${r.domain}`).join(", ")
          : `${d.error_kind}, continuing without web evidence (${d.message_safe})`) + `, ${d.latency_ms} ms`,
        d.status === "ok" ? "mcp" : "warn");
    },
    retrieval: (d) => {
      d.chunks.forEach((c) => registry.set(c.cite_id, { ...c, label: `${c.title} — ${c.section}` }));
      traceItem(turn, "retrieval", ` "${d.query}": ${d.chunks.map((c) => `${c.cite_id} ${c.doc_id} (${c.score.toFixed(2)})`).join(", ")} in ${d.retrieval_ms} ms`);
    },
    sql_result: (d) => {
      (d.refs || []).forEach((r) => registry.set(r.cite_id, {
        ...r, sql: d.sql, columns: d.columns, source_url: r.href, agent: d.agent,
        label: `${r.title} — corpus metadata (SQL)`,
      }));
      traceItem(turn, `${d.agent} ⇄ SQLite`, d.status === "ok"
        ? ` run_sql (read-only) ${d.sql} → ${d.n_rows} row(s)${d.truncated ? " (capped at 50)" : ""} in ${d.elapsed_ms} ms: ` +
          (d.cite_ids.length ? d.cite_ids.join(" ") : "none")
        : ` run_sql ${d.error_kind}: ${d.message_safe} · SQL: ${d.sql}` +
          (d.repair_rounds_left ? " · one repair round left" : " · no repair rounds left"),
        d.status === "ok" ? "mcp" : "warn");
    },
    handoff: (d) => traceItem(turn, "handoff", ` ${d.from_tasks.join(", ")} → ${d.to_task} ${d.to_agent}: ` +
      (d.papers.length ? d.papers.map((p) => `arXiv:${p.arxiv_id}${p.cite_id ? ` [${p.cite_id}]` : ""}`).join(", ")
        : "no paper ids") + (d.evidence_ids.length ? ` · upstream evidence ${d.evidence_ids.join(" ")}` : "")),
    worker_result: (d) => traceItem(turn, d.agent, ` ${d.task_id} ${d.status}` +
      (d.attempt > 1 ? ` (attempt ${d.attempt})` : "") +
      (d.error_kind ? ` (${d.error_kind})` : "") + (d.reason ? `: ${d.reason}` : "") + INJ(d) +
      (d.evidence_ids && d.evidence_ids.length ? `, cites ${d.evidence_ids.join(" ")}` : ""),
      statusKind(d.status)),
    token: (d) => { text += d.delta; repaint(); },
    citation: (d) => {
      const ref = { ...(registry.get(d.cite_id) || {}), ...d }; // keeps the web snippet from web_search
      registry.set(d.cite_id, ref);
      const li = el("li");
      const a = el("a", "cite", d.cite_id);
      a.href = "#";
      a.addEventListener("click", (e) => { e.preventDefault(); openSource(ref); });
      li.append(el("span", "badge", `#${turn}`), a, document.createTextNode(
        d.type === "web" ? ` ${d.label} (${d.doc_id})` : d.type === "sql_result" ? ` ${d.label}` : ` ${d.label} (arXiv:${d.doc_id})`));
      $("sources").append(li);
      repaint();
    },
    error: (d) => {
      traceItem(turn, "error", ` ${d.scope}${d.agent ? " " + d.agent : ""}: ${d.message_safe}${INJ(d)}`, d.recoverable ? "warn" : "bad");
      if (!d.recoverable) box.append(el("div", "status error", `Error: ${d.message_safe}`));
    },
    done: (d) => {
      gotDone = true;
      const t = d.timings;
      const tok = Object.entries(d.tokens_by_provider).map(([p, v]) => `${SHORT[p] || p} ${v.calls} calls, ${v.input_tokens}→${v.output_tokens} tok`).join("; ") +
        (d.cost_usd != null ? ` · cost $${d.cost_usd.toFixed(4)}` +
          (d.cost_basis && d.cost_basis.estimated ? ` (${d.cost_basis.estimated} call(s) estimated)` : "") : "") + (d.fallbacks ? ` · ${d.fallbacks} fallback(s)` : "");
      status.textContent = `${d.status} · TTFT ${t.ttft_ms ?? "–"} ms · total ${t.total_ms} ms · retrieval ${t.retrieval_ms} ms · ${tok}` +
        (d.unresolved_citations.length ? ` · unresolved: ${d.unresolved_citations.join(" ")}` : "") +
        (d.dropped_citations && d.dropped_citations.length ? ` · not linked (missing-evidence statements): ${d.dropped_citations.join(" ")}` : "");
      d.dropped_citations && d.dropped_citations.forEach((c) => dropped.add(c));
      repaint();
      if (d.status !== "done") status.classList.add("error");
      traceItem(turn, "done", ` ${d.status} in ${t.total_ms} ms` +
        (d.tasks && d.tasks.length ? " · " + d.tasks.map((x) => `${x.id} ${x.worker} ${x.status}`).join(", ") : "") +
        (d.faults_injected && d.faults_injected.length ? ` · ${d.faults_injected.length} injected fault(s)` : ""), statusKind(d.status));
    },
  };

  const parse = makeParser((ev, data) => (handlers[ev] || (() => {}))(data));
  try {
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream",
        ...(FAULT ? { "X-Fault-Inject": FAULT.spec, "X-Fault-Token": FAULT.token } : {}) },
      body: JSON.stringify({ query }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${(await r.text()).slice(0, 200)}`);
    const reader = r.body.getReader();
    const dec = new TextDecoder("utf-8"); // stream mode keeps multi-byte characters split across chunks intact
    for (;;) {
      const { value, done } = await reader.read();
      if (done) { parse(dec.decode(), true); break; }
      parse(dec.decode(value, { stream: true }));
    }
    if (!gotDone) throw new Error("the connection closed before the answer finished");
  } catch (e) {
    box.append(el("div", "status error", `Error: ${e.message}`));
  } finally {
    answer.classList.remove("streaming");
    renderAnswer(answer, text, registry, dropped);
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
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("ask").requestSubmit(); }
});
