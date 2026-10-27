// Tester Phase 2 (T2-JS): run the real makeParser() from app/static/app.js under Node with
// arbitrary network splits, multi-byte characters split across chunks, CRLF framing and keep-alives.
import { readFileSync } from "node:fs";
const src = readFileSync(process.argv[2], "utf8");
const start = src.indexOf("function makeParser(");
const end = src.indexOf("\n}\n", start) + 2;
const makeParser = new Function(src.slice(start, end) + "\nreturn makeParser;")();

const payloads = [
  ["meta", { request_id: "r1", seq: 1 }],
  ["token", { delta: "naïve — 检索 😀\nline2 \"q\" data: event: x" }],
  ["token", { delta: "\r\n\r\n not a frame" }],
  ["citation", { cite_id: "C1", label: "Title — §3" }],
  ["done", { status: "done" }],
];
let wire = "";
payloads.forEach(([e, d], i) => {
  wire += (i === 2 ? ": ping\n\n" : "") + `event: ${e}\ndata: ${JSON.stringify(d)}\n\n`;
});
const expected = JSON.stringify(payloads);
const bytes = new TextEncoder().encode(wire);
let failures = 0, runs = 0;
function runSplit(cuts, crlf = false) {
  const b = crlf ? new TextEncoder().encode(wire.replace(/\n/g, "\r\n")) : bytes;
  const got = [];
  const parse = makeParser((ev, data) => got.push([ev, data]));
  const dec = new TextDecoder("utf-8");
  let prev = 0;
  for (const c of [...cuts, b.length]) {
    parse(dec.decode(b.slice(prev, c), { stream: true }));
    prev = c;
  }
  parse(dec.decode(), true);
  runs++;
  if (JSON.stringify(got) !== expected) { failures++; if (failures < 3) console.error("MISMATCH", cuts, JSON.stringify(got).slice(0, 300)); }
}
// every single split point, then random multi-splits, then 1-byte chunks
for (let i = 1; i < bytes.length; i++) runSplit([i]);
let seed = 7; const rnd = () => (seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31;
for (let t = 0; t < 2000; t++) {
  const n = 1 + Math.floor(rnd() * 12);
  const cuts = [...new Set(Array.from({ length: n }, () => 1 + Math.floor(rnd() * (bytes.length - 1))))].sort((a, b) => a - b);
  runSplit(cuts);
}
runSplit(Array.from({ length: bytes.length - 1 }, (_, i) => i + 1));
// CRLF framing (the parser claims \r?\n support)
const crlfLen = new TextEncoder().encode(wire.replace(/\n/g, "\r\n")).length;
for (let i = 1; i < crlfLen; i += 3) runSplit([i], true);
console.log(JSON.stringify({ runs, failures, wire_bytes: bytes.length }));
process.exit(failures ? 1 : 0);
