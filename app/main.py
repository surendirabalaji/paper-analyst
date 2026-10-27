"""FastAPI app: SSE chat endpoint, source inspection, health, and the static chat page."""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import ROOT, load_settings
from app.context import RequestContext
from app.events import Emitter, sse_frame
from app.graph import run_request
from app.llm.router import Router
from app.rag import MAX_K, MIN_K, Retriever

STATIC = ROOT / "app" / "static"
KEEPALIVE_S = 15

# httpx logs full request URLs at INFO; keep them out of the logs (credentials may sit in query strings).
for name in ("httpx", "httpcore"):
    logging.getLogger(name).setLevel(logging.WARNING)
log = logging.getLogger("paper_analyst")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    app.state.settings = settings
    app.state.retriever = Retriever(settings)  # refuses to start on an embedding-model mismatch
    app.state.router = Router(settings)
    log.info("index ready: %s chunks, %s-d", app.state.retriever.count, app.state.retriever.dims)
    yield


app = FastAPI(title="Agentic AI Paper Analyst", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k: int | None = Field(None, ge=MIN_K, le=MAX_K)


@app.post("/api/chat")
async def chat(body: ChatRequest, request: Request):
    query = body.query.strip()
    if not query:
        raise HTTPException(422, "query is empty")
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
    return {"status": "ok", "chunks": r.count, "embedding_dims": r.dims}


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")
