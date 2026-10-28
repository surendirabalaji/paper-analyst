"""FastAPI app: SSE chat endpoint, source inspection, health, and the static chat page."""

from __future__ import annotations

import asyncio
import hmac
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import ROOT, load_settings
from app.context import RequestContext
from app.events import Emitter, RedactingFilter, sse_frame
from app.faults import Faults, FaultSpecError
from app.graph import run_request
from app.llm.router import Router
from app.mcp_tavily import TavilyMCP
from app.rag import MAX_K, MIN_K, Retriever

STATIC = ROOT / "app" / "static"
KEEPALIVE_S = 15
FAULT_HEADER, FAULT_TOKEN_HEADER = "x-fault-inject", "x-fault-token"
MIN_FAULT_TOKEN_CHARS = 24

# httpx logs full request URLs at INFO and the MCP client logs session details; the Tavily key sits in the
# MCP URL's query string, so keep these loggers at WARNING. RedactingFilter is a backstop on every handler.
for name in ("httpx", "httpcore", "mcp", "langchain_mcp_adapters"):
    logging.getLogger(name).setLevel(logging.WARNING)
log = logging.getLogger("paper_analyst")
if not log.handlers:  # uvicorn only configures its own loggers; send ours to stderr (Cloud Run captures it)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s:     %(name)s: %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False


def install_redaction() -> None:
    """Attach RedactingFilter to our handler, the root handlers and uvicorn's handlers (idempotent)."""
    loggers = [log, logging.getLogger(), *(logging.getLogger(n) for n in ("uvicorn", "uvicorn.error", "uvicorn.access"))]
    for lg in loggers:
        for h in lg.handlers:
            if not any(isinstance(f, RedactingFilter) for f in h.filters):
                h.addFilter(RedactingFilter())


install_redaction()


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    install_redaction()  # uvicorn has configured its handlers by now
    settings = load_settings()
    app.state.settings = settings
    app.state.retriever = Retriever(settings)  # refuses to start on an embedding-model mismatch
    app.state.router = Router(settings)
    app.state.tavily = TavilyMCP(settings)
    if settings.fault_inject:
        Faults.from_spec(settings.fault_inject, "env")  # a bad spec fails at startup, not per request
        log.warning("FAULT_INJECT is set: every request gets injected test faults (%s)", settings.fault_inject)
    token_ok = len(settings.fault_inject_token) >= MIN_FAULT_TOKEN_CHARS
    if settings.fault_inject_token and not token_ok:
        log.warning("FAULT_INJECT_TOKEN is shorter than %d characters; per-request fault injection stays off",
                    MIN_FAULT_TOKEN_CHARS)
    app.state.fault_token = settings.fault_inject_token if token_ok else ""
    log.info("fault injection: per-request %s", "enabled for requests with the token" if token_ok else "off")
    log.info(
        "index ready: %s chunks, %s-d, live search %s, startup %.2fs",
        app.state.retriever.count, app.state.retriever.dims,
        "enabled" if app.state.tavily.enabled else "DISABLED (no TAVILY_API_KEY)", time.perf_counter() - t0,
    )
    yield


app = FastAPI(title="Agentic AI Paper Analyst", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k: int | None = Field(None, ge=MIN_K, le=MAX_K)


def request_faults(request: Request) -> Faults:
    """Fault rules for this request. Ordinary requests (no X-Fault-Inject header) get none on the deployed
    service. The header is honoured only if the server has a fault token and X-Fault-Token matches it
    (constant-time compare); otherwise the request is refused, so a fault can never be triggered by accident
    or by an ordinary user. FAULT_INJECT (process-wide) is for local runs and is never deployed."""
    s = request.app.state
    spec = request.headers.get(FAULT_HEADER)
    if spec is None:
        return Faults.from_spec(s.settings.fault_inject, "env") if s.settings.fault_inject else Faults()
    token = getattr(s, "fault_token", "")
    given = request.headers.get(FAULT_TOKEN_HEADER, "")
    if not token or not hmac.compare_digest(given.encode(), token.encode()):
        raise HTTPException(403, "fault injection is not enabled for this request")
    try:
        return Faults.from_spec(spec, "header")
    except FaultSpecError as e:
        raise HTTPException(422, f"bad {FAULT_HEADER} header: {e}") from None


@app.post("/api/chat")
async def chat(body: ChatRequest, request: Request):
    query = body.query.strip()
    if not query:
        raise HTTPException(422, "query is empty")
    faults = request_faults(request)
    s = request.app.state
    request_id = uuid.uuid4().hex[:12]
    ctx = RequestContext(
        request_id=request_id,
        query=query,
        k=body.k or s.settings.top_k,
        settings=s.settings,
        router=s.router,
        retriever=s.retriever,
        emitter=Emitter(request_id),
        tavily=s.tavily,
        faults=faults,
    )

    async def stream():
        task = asyncio.create_task(run_request(ctx))
        try:
            while True:
                try:
                    item = await ctx.emitter.next(KEEPALIVE_S)
                except StopAsyncIteration:
                    break
                if item is None:
                    if await request.is_disconnected():
                        break
                    yield ": ping\n\n"
                    continue
                yield sse_frame(*item)
        finally:
            if not task.done():  # client went away: stop the graph and its provider calls
                task.cancel()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Request-Id": request_id},
    )


@app.get("/api/source/chunk/{chunk_id}")
async def source_chunk(chunk_id: str, request: Request):
    # Only ids that exist in the vector index resolve; nothing here touches the filesystem by path.
    found = request.app.state.retriever.get_chunk(chunk_id)
    if not found:
        raise HTTPException(404, "unknown chunk id")
    return found


@app.get("/health")
async def health(request: Request):
    r = request.app.state.retriever
    return {"status": "ok", "chunks": r.count, "embedding_dims": r.dims,
            "live_search": "tavily-remote-mcp" if request.app.state.tavily.enabled else "disabled"}


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")
