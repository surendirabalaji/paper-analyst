"""Per-request SSE event emitter.

Graph nodes push events into a request-local asyncio.Queue; the HTTP handler drains it and writes
SSE frames. Every payload carries request_id, seq and ts so concurrent requests stay distinguishable.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from typing import Any

_DONE = object()


_SECRET_PATTERNS = [
    (re.compile(r"tavilyApiKey(=|%3D)[^&\s'\"]+", re.I), r"tavilyApiKey\1REDACTED"),
    (re.compile(r"\b(sk-[A-Za-z0-9_-]{6})[A-Za-z0-9_-]+"), r"\1…REDACTED"),
    (re.compile(r"\btvly-[A-Za-z0-9_-]+"), "tvly-REDACTED"),
    (re.compile(r"(Bearer\s+)\S+", re.I), r"\1REDACTED"),
]


def safe_error(e: BaseException, limit: int = 300) -> str:
    """Exception text that is safe to show in the trace: credentials redacted, length capped."""
    s = f"{type(e).__name__}: {e}"
    for pat, rep in _SECRET_PATTERNS:
        s = pat.sub(rep, s)
    return s[:limit]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sse_frame(event: str, payload: dict[str, Any]) -> str:
    # json.dumps never emits raw newlines, so a single data: line is a complete, valid frame.
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


class Emitter:
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.queue: asyncio.Queue = asyncio.Queue()
        self.seq = 0
        self.t0 = time.perf_counter()
        self.closed = False

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.t0) * 1000)

    def emit(self, event: str, **data: Any) -> None:
        if self.closed:
            return
        self.seq += 1
        payload = {"request_id": self.request_id, "seq": self.seq, "ts": utc_now(), "t_ms": self.elapsed_ms()}
        payload.update(data)
        self.queue.put_nowait((event, payload))

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.queue.put_nowait(_DONE)

    async def next(self, timeout: float):
        """Next (event, payload); None on timeout (caller sends a keep-alive); raises StopAsyncIteration at end."""
        try:
            item = await asyncio.wait_for(self.queue.get(), timeout)
        except asyncio.TimeoutError:
            return None
        if item is _DONE:
            raise StopAsyncIteration
        return item
