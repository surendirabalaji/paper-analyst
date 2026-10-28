// Minimal Markdown parser for model answers: text in, a plain-data tree out. It never produces HTML; app.js
// builds DOM nodes from the tree with createElement/textContent, so model output cannot inject markup.
// Supported: # headings, ---, - / 1. lists (one nesting level), > quotes, ``` fences, | tables |,
// **bold**, *italic*, `code` and [text](http(s) links). Anything else stays literal text. Partial Markdown
// (mid-stream) degrades to text and is re-parsed on the next repaint.
"use strict";

const MD = (() => {
  const HEADING = /^(#{1,6})\s+(.*?)\s*#*\s*$/;
  const HR = /^\s{0,3}([-*_])(?:\s*\1){2,}\s*$/;
  const UL = /^(\s*)[-*+]\s+(.*)$/;
  const OL = /^(\s*)(\d{1,9})[.)]\s+(.*)$/;
  const QUOTE = /^\s{0,3}>\s?(.*)$/;
  const FENCE = /^\s{0,3}(```|~~~)\s*([\w+-]*)\s*$/;
  const TABLE_SEP = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;
  const INLINE = /`([^`\n]+)`|\*\*(?=\S)([\s\S]*?\S)\*\*|__(?=\S)([\s\S]*?\S)__|\*(?=[^\s*])([^*\n]*?[^\s*])\*|\[([^\]\n]+)\]\((https?:\/\/[^\s()]+)\)/g;

  function inline(text, depth = 0) {
    const out = [];
    let last = 0;
    for (const m of text.matchAll(INLINE)) {
      if (m.index > last) out.push({ t: "text", v: text.slice(last, m.index) });
      const nest = (s) => (depth < 4 ? inline(s, depth + 1) : [{ t: "text", v: s }]);
      if (m[1] !== undefined) out.push({ t: "code", v: m[1] });
      else if (m[2] !== undefined) out.push({ t: "strong", c: nest(m[2]) });
      else if (m[3] !== undefined) out.push({ t: "strong", c: nest(m[3]) });
      else if (m[4] !== undefined) out.push({ t: "em", c: nest(m[4]) });
      else out.push({ t: "link", href: m[6], c: nest(m[5]) });
      last = m.index + m[0].length;
    }
    if (last < text.length) out.push({ t: "text", v: text.slice(last) });
    return out;
  }

  const cells = (line) => line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => inline(c.trim()));
  const isListLine = (l) => UL.test(l) || OL.test(l);

  function parse(src) {
    const lines = String(src).replace(/\r\n?/g, "\n").split("\n");
    const blocks = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) { i++; continue; }
      let m;
      if ((m = line.match(FENCE))) { // fenced code: until the closing fence (or the end, mid-stream)
        const body = [];
        i++;
        while (i < lines.length && !(lines[i].trim().startsWith(m[1]))) body.push(lines[i++]);
        i++;
        blocks.push({ t: "code", lang: m[2] || "", v: body.join("\n") });
        continue;
      }
      if ((m = line.match(HEADING))) { blocks.push({ t: "h", level: m[1].length, c: inline(m[2]) }); i++; continue; }
      if (HR.test(line)) { blocks.push({ t: "hr" }); i++; continue; } // before lists: "- - -" is a rule
      if (line.includes("|") && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1]) && lines[i + 1].includes("-")) {
        const head = cells(line);
        const rows = [];
        i += 2;
        while (i < lines.length && lines[i].includes("|") && lines[i].trim()) rows.push(cells(lines[i++]));
        blocks.push({ t: "table", head, rows });
        continue;
      }
      if (QUOTE.test(line)) {
        const body = [];
        while (i < lines.length && (m = lines[i].match(QUOTE))) { body.push(m[1]); i++; }
        blocks.push({ t: "quote", c: parse(body.join("\n")) });
        continue;
      }
      if (isListLine(line)) {
        blocks.push(list(lines, i, (n) => (i = n)));
        continue;
      }
      const para = [];
      while (i < lines.length && lines[i].trim() && !HEADING.test(lines[i]) && !FENCE.test(lines[i]) &&
             !HR.test(lines[i]) && !isListLine(lines[i]) && !QUOTE.test(lines[i])) para.push(lines[i++].trim());
      blocks.push({ t: "p", c: inline(para.join("\n")) });
    }
    return blocks;
  }

  // A list starting at lines[i]; items indented deeper than the first item become one nested list.
  function list(lines, i, setI) {
    const first = lines[i].match(UL) || lines[i].match(OL);
    const ordered = !lines[i].match(UL);
    const base = first[1].length;
    const node = { t: ordered ? "ol" : "ul", start: ordered ? Number(first[2]) : 1, items: [] };
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) { // a blank line ends the list unless another item of it follows
        if (i + 1 < lines.length && isListLine(lines[i + 1]) &&
            (lines[i + 1].match(UL) || lines[i + 1].match(OL))[1].length >= base) { i++; continue; }
        break;
      }
      const m = line.match(UL) || line.match(OL);
      if (HR.test(line)) break;
      if (!m) { // continuation text of the previous item
        if (!node.items.length || HEADING.test(line) || HR.test(line) || FENCE.test(line)) break;
        node.items[node.items.length - 1].c.push({ t: "text", v: "\n" }, ...inline(line.trim()));
        i++;
        continue;
      }
      const indent = m[1].length;
      if (indent < base) break;
      if (indent > base && node.items.length) {
        node.items[node.items.length - 1].sub = list(lines, i, (n) => (i = n));
        continue;
      }
      if (!!line.match(UL) === ordered) break; // switching list type at the same level starts a new list
      node.items.push({ c: inline(ordered ? m[3] : m[2]) });
      i++;
    }
    setI(i);
    return node;
  }

  return { parse, inline };
})();

if (typeof module !== "undefined") module.exports = MD;
