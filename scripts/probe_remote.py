#!/usr/bin/env python3
"""Time one /health call and one streamed /api/chat question against a deployed (or local) URL.

Client-side wall-clock times: health latency, time to the first SSE byte, time to the first `token`
event (TTFT), and total. Prints one JSON line; nothing secret is sent or printed.

Usage: .venv/bin/python scripts/probe_remote.py https://SERVICE-URL ["question"]
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone

import httpx

DEFAULT_Q = "How does Self-RAG decide when to retrieve?"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    url = sys.argv[1].rstrip("/")
    query = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_Q
    out: dict = {"url": url, "query": query, "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    timeout = httpx.Timeout(300.0, connect=60.0)
    with httpx.Client(timeout=timeout) as client:
        t0 = time.perf_counter()
        h = client.get(f"{url}/health")
        out["health_s"] = round(time.perf_counter() - t0, 3)
        out["health_status"] = h.status_code

        counts: dict[str, int] = {}
        errors: list[str] = []
        first_byte = first_token = None
        event = None
        t0 = time.perf_counter()
        with client.stream("POST", f"{url}/api/chat", json={"query": query}) as r:
            out["chat_status"] = r.status_code
            out["request_id"] = r.headers.get("x-request-id")
            for line in r.iter_lines():
                if first_byte is None:
                    first_byte = time.perf_counter() - t0
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    counts[event] = counts.get(event, 0) + 1
                    if event == "token" and first_token is None:
                        first_token = time.perf_counter() - t0
                    if event == "error":
                        d = json.loads(line[5:])
                        errors.append(f"{d.get('scope')}/{d.get('agent', '-')}: {d.get('message_safe', '')}"[:200])
        out["chat_first_byte_s"] = round(first_byte or -1, 3)
        out["chat_ttft_s"] = round(first_token or -1, 3)
        out["chat_total_s"] = round(time.perf_counter() - t0, 3)
        out["events"] = counts
        out["errors"] = errors
    out["ok"] = (
        out["health_status"] == 200 and out["chat_status"] == 200 and counts.get("token", 0) > 0
        and counts.get("citation", 0) > 0 and counts.get("done") == 1 and not errors
    )
    print(json.dumps(out))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
