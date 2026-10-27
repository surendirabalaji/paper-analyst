// Paper Analyst chat client: POST + fetch() streaming, incremental SSE parsing, no build step.
// All server text is inserted with textContent; links are built with DOM APIs only for known citation ids.
"use strict";

const $ = (id) => document.getElementById(id);
const MARKERS = /\[([A-Z]\d+(?:\s*[,;]\s*[A-Z]\d+)*)\]/g;
let turnNo = 0;

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

function renderAnswer(node, text, registry) {
  node.replaceChildren();
  let last = 0;
  for (const m of text.matchAll(MARKERS)) {
    node.append(document.createTextNode(text.slice(last, m.index)));
    node.append(document.createTextNode("["));
    m[1].split(/\s*[,;]\s*/).forEach((id, i) => {
      if (i) node.append(document.createTextNode(", "));
      const ref = registry.get(id);
      if (ref) {
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

async function openSource(ref) {
  showTab("sources");
  const v = $("viewer");
  v.hidden = false;
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
  const routes = new Map();
  let text = "";
  let gotDone = false;
  let pending = false;
  const repaint = () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; renderAnswer(answer, text, registry); });
  };

  const handlers = {
    meta: (d) => traceItem(turn, "start", ` request ${d.request_id}, k=${d.k}, workers: ${d.workers.join(", ")}`),
    plan: (d) => {
      traceItem(turn, "plan", ` ${d.plan_summary} (${d.source})`);
      d.tasks.forEach((t) => traceItem(turn, t.id, ` ${t.worker}: ${t.instruction}` +
        (t.depends_on.length ? ` (after ${t.depends_on.join(", ")})` : "")));
      d.notes.forEach((n) => traceItem(turn, "note", ` ${n}`, "warn"));
      status.textContent = d.tasks.length ? `Plan: ${d.tasks.map((t) => t.worker).join(" → ")}` : "No research needed";
    },
    transition: (d) => {
      traceItem(turn, "→", ` ${d.from} → ${d.to}: ${d.reason}`);
      if (d.to !== "supervisor") status.textContent = `${d.to}…`;
    },
    llm_route: (d) => {
      const tr = el("tr");
      tr.append(el("td", "", `#${turn}`), el("td", "", `${d.agent} / ${d.call_site}`),
        el("td", "", `${d.provider_label} · ${d.model_id}`), el("td", "", `${d.dimension}: ${d.rationale}`),
        el("td", "", "…"));
      $("routing-rows").append(tr);
      routes.set(d.decision_id, tr);
    },
    llm_result: (d) => {
      const tr = routes.get(d.decision_id);
      if (tr) tr.lastChild.textContent = `${d.input_tokens}→${d.output_tokens} · ${d.latency_ms} ms` +
        (d.ttft_ms != null ? ` (TTFT ${d.ttft_ms})` : "");
    },
    tool_call: (d) => traceItem(turn, d.agent, ` calls ${d.tool}(${JSON.stringify(d.args_safe)})`),
    tool_result: (d) => traceItem(turn, d.agent, ` ${d.tool} → ${d.status}` +
      (d.n_results != null ? `, ${d.n_results} results` : "") + `, ${d.latency_ms} ms`, statusKind(d.status)),
    retrieval: (d) => {
      d.chunks.forEach((c) => registry.set(c.cite_id, { ...c, label: `${c.title} — ${c.section}` }));
      traceItem(turn, "retrieval", ` "${d.query}": ${d.chunks.map((c) => `${c.cite_id} ${c.doc_id} (${c.score.toFixed(2)})`).join(", ")} in ${d.retrieval_ms} ms`);
    },
    worker_result: (d) => traceItem(turn, d.agent, ` ${d.task_id} ${d.status}` +
      (d.error_kind ? ` (${d.error_kind})` : "") + (d.evidence_ids && d.evidence_ids.length ? `, cites ${d.evidence_ids.join(" ")}` : ""),
      statusKind(d.status)),
    token: (d) => { text += d.delta; repaint(); },
    citation: (d) => {
      registry.set(d.cite_id, { ...d });
      const li = el("li");
      const a = el("a", "cite", d.cite_id);
      a.href = "#";
      a.addEventListener("click", (e) => { e.preventDefault(); openSource(d); });
      li.append(el("span", "badge", `#${turn}`), a, document.createTextNode(` ${d.label} (arXiv:${d.doc_id})`));
      $("sources").append(li);
      repaint();
    },
    error: (d) => {
      traceItem(turn, "error", ` ${d.scope}${d.agent ? " " + d.agent : ""}: ${d.message_safe}`, d.recoverable ? "warn" : "bad");
      if (!d.recoverable) box.append(el("div", "status error", `Error: ${d.message_safe}`));
    },
    done: (d) => {
      gotDone = true;
      const t = d.timings;
      const tok = Object.entries(d.tokens_by_provider).map(([p, v]) => `${p} ${v.calls} calls, ${v.input_tokens}→${v.output_tokens} tok`).join("; ");
      status.textContent = `${d.status} · TTFT ${t.ttft_ms ?? "–"} ms · total ${t.total_ms} ms · retrieval ${t.retrieval_ms} ms · ${tok}` +
        (d.unresolved_citations.length ? ` · unresolved: ${d.unresolved_citations.join(" ")}` : "");
      if (d.status !== "done") status.classList.add("error");
      traceItem(turn, "done", ` ${d.status} in ${t.total_ms} ms`, statusKind(d.status));
    },
  };

  const parse = makeParser((ev, data) => (handlers[ev] || (() => {}))(data));
  try {
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
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
    renderAnswer(answer, text, registry);
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
