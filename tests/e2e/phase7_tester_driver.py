"""Tester Phase 7 driver: POST /api/chat with an optional fault spec, save redacted SSE events + a summary.

Usage: python tests/e2e/phase7_tester_driver.py BASE_URL OUT_JSONL "QUERY" [SPEC] [--token=dotenv|env|none|wrong]
The token is read from .env (FAULT_INJECT_TOKEN) or the env var P7_TOKEN; it is never printed or saved, and the
saved file is checked not to contain it. Prints one JSON summary line.
"""
import json, os, pathlib, sys, time
import httpx
from dotenv import dotenv_values

ROOT = pathlib.Path(__file__).resolve().parents[2]
args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
base, out, query = args[0].rstrip("/"), pathlib.Path(args[1]), args[2]
spec = args[3] if len(args) > 3 else None
mode = opts.get("token", "dotenv")
token = {"dotenv": lambda: dotenv_values(ROOT / ".env").get("FAULT_INJECT_TOKEN") or "",
         "env": lambda: os.environ.get("P7_TOKEN", ""), "none": lambda: None,
         "wrong": lambda: "W" * 43}[mode]()
h = {}
if spec is not None:
    h["X-Fault-Inject"] = spec
if token is not None and spec is not None:
    h["X-Fault-Token"] = token
t0 = time.perf_counter()
events, first_tok, status_code = [], None, None
raw_all = ""
with httpx.stream("POST", f"{base}/api/chat", json={"query": query}, headers=h, timeout=httpx.Timeout(330, connect=60)) as r:
    status_code = r.status_code
    buf = ""
    for chunk in r.iter_text():
        raw_all += chunk
        buf += chunk
        while "\n\n" in buf:
            fr, buf = buf.split("\n\n", 1)
            e = d = None
            for ln in fr.splitlines():
                if ln.startswith("event: "): e = ln[7:]
                elif ln.startswith("data: "): d = json.loads(ln[6:])
            if e:
                t = round(time.perf_counter() - t0, 3)
                if e == "token" and first_tok is None: first_tok = t
                events.append({"t_client_s": t, "event": e, "data": d})
wall = round(time.perf_counter() - t0, 2)
leak = bool(token) and token in raw_all
out.parent.mkdir(parents=True, exist_ok=True)
if not leak:
    out.write_text("".join(json.dumps(x, default=str) + "\n" for x in events) if events else raw_all)
def g(n): return [x["data"] for x in events if x["event"] == n]
done = (g("done") or [{}])[-1]
ans = "".join(d.get("delta", "") for d in g("token"))
print(json.dumps({"http": status_code, "wall_s": wall, "first_token_s": first_tok, "n_events": len(events),
    "status": done.get("status"), "tasks": done.get("tasks"), "faults": [f.get("target") for f in done.get("faults_injected", [])],
    "fault_events": len(g("fault_injected")), "providers_used": done.get("providers_used"), "fallbacks": done.get("fallbacks"),
    "cost": done.get("cost_usd"), "over_budget_s": done.get("over_budget_s"), "citations": [c.get("cite_id") for c in g("citation")],
    "errors": [{k: e.get(k) for k in ("scope", "agent", "error_kind", "recoverable", "injected")} for e in g("error")],
    "token_leak": leak, "answer_tail": ans[-420:], "body_if_not_sse": raw_all[:200] if not events else None}, default=str))
