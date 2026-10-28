// Phase 8: app/static/md.js (the answer Markdown parser) under Node. It must return plain data only
// (no HTML strings), keep HTML-looking model text as literal text, and keep Unicode intact.
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const MD = require(process.argv[2]);
let fails = 0;
const ok = (c, msg, got) => { if (!c) { fails++; console.error("FAIL", msg, JSON.stringify(got).slice(0, 400)); } };
const texts = (nodes) => nodes.map((n) => (n.t === "text" || n.t === "code" ? n.v : texts(n.c || []))).join("");
const allText = (blocks) => blocks.map((b) => b.c ? (Array.isArray(b.c) && b.c[0] && b.c[0].t ? texts(b.c) : allText(b.c))
  : b.items ? b.items.map((i) => texts(i.c) + (i.sub ? allText([i.sub]) : "")).join("|")
  : b.t === "code" ? b.v : b.t === "table" ? [...b.head, ...b.rows.flat()].map(texts).join("|") : "").join("\n");

// 1. block types
let b = MD.parse("## Findings\n\nReAct **interleaves** reasoning [C1] and `act()` steps.\n\n---\n\n- one [C2]\n- two *em*\n  - nested\n\n1. first\n2. second\n\n> quoted\n\n```py\nx = '<b>'\n```\n\n| a | b |\n|---|---|\n| 1 | [D1] |");
ok(b.map((x) => x.t).join(",") === "h,p,hr,ul,ol,quote,code,table", "block kinds", b.map((x) => x.t));
ok(b[0].level === 2 && texts(b[0].c) === "Findings", "heading", b[0]);
ok(b[1].c.some((n) => n.t === "strong" && texts(n.c) === "interleaves"), "bold", b[1]);
ok(b[1].c.some((n) => n.t === "code" && n.v === "act()"), "inline code", b[1]);
ok(texts(b[1].c).includes("[C1]"), "citation marker kept as text for app.js", b[1]);
ok(b[3].items.length === 2 && b[3].items[1].sub && b[3].items[1].sub.items.length === 1, "nested list", b[3]);
ok(b[3].items[1].c.some((n) => n.t === "em"), "italic", b[3]);
ok(b[4].t === "ol" && b[4].items.length === 2, "ordered list", b[4]);
ok(b[6].v === "x = '<b>'" && b[6].lang === "py", "fence", b[6]);
ok(b[7].head.length === 2 && b[7].rows.length === 1, "table", b[7]);

// 2. HTML in model output stays literal text; there is no node type that carries raw HTML
const evil = '<img src=x onerror=alert(1)> <script>alert(2)</script> **<b>x</b>** [click](javascript:alert(3)) [ok](https://arxiv.org/abs/2210.03629)';
b = MD.parse(evil);
const json = JSON.stringify(b);
ok(!/"t":"(html|raw)"/.test(json), "no html node type", b);
ok(texts(b[0].c).includes("<img src=x onerror=alert(1)>") && texts(b[0].c).includes("<script>"), "html kept as text", b);
ok(!b[0].c.some((n) => n.t === "link" && !/^https?:/.test(n.href)), "only http(s) links", b);
ok(b[0].c.some((n) => n.t === "link" && n.href === "https://arxiv.org/abs/2210.03629"), "https link parsed", b);
const types = new Set();
(function walk(x) { if (Array.isArray(x)) x.forEach(walk); else if (x && typeof x === "object") { if (x.t) types.add(x.t); Object.values(x).forEach(walk); } })(b);
ok([...types].every((t) => ["p", "text", "strong", "em", "code", "link"].includes(t)), "node types whitelist", [...types]);

// 3. Unicode: accents, CJK, emoji, RTL survive exactly
const uni = "## Résumé — 检索增强生成 😀\n\n**naïve** 日本語の*テスト* [C1] مرحبا 👩‍🔬";
b = MD.parse(uni);
ok(texts(b[0].c) === "Résumé — 检索增强生成 😀", "unicode heading", b[0]);
ok(texts(b[1].c) === "naïve 日本語のテスト [C1] مرحبا 👩‍🔬", "unicode inline", b[1]);

// 4. partial Markdown mid-stream never throws and keeps all text
for (const s of ["**bold not closed", "```\ncode not closed", "- item\n-", "| a | b |\n|--", "# ", "*", "`", "1."]) {
  let r;
  try { r = MD.parse(s); } catch (e) { ok(false, "threw on " + s, String(e)); continue; }
  ok(Array.isArray(r), "array for " + s, r);
}
ok(texts(MD.parse("**bold not closed")[0].c) === "**bold not closed", "unclosed bold stays literal", 0);

// 5. paragraph lines keep their line breaks; a gaps note after --- renders as hr + bold
b = MD.parse("Answer line one\nline two\n\n---\n**Note:** the Data task failed [injected test fault].");
ok(b[0].t === "p" && texts(b[0].c) === "Answer line one\nline two", "line breaks kept", b[0]);
ok(b[1].t === "hr" && b[2].c[0].t === "strong", "gaps note", b);

// 6. bounded work on adversarial input (no catastrophic backtracking)
const t0 = Date.now();
MD.parse("*".repeat(20000) + "a" + "_".repeat(20000) + "`".repeat(5000) + "[".repeat(5000) + "|".repeat(5000));
MD.parse(("- a\n  - b\n".repeat(2000)) + ("> q\n".repeat(2000)));
ok(Date.now() - t0 < 2000, "adversarial input fast", Date.now() - t0);

console.log(JSON.stringify({ failures: fails }));
process.exit(fails ? 1 : 0);
