"""Tester-owned fakes for Phase 2 offline tests (no network, no paid calls).

FakeChat is a real LangChain BaseChatModel, so the application's own code paths
(with_structured_output, bind_tools, AUGMENT_PROMPT | model, astream, ToolMessage handling) execute
unchanged; only the provider response is scripted. Results from these fakes are MOCKED evidence.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from app.config import Settings
from app.context import RequestContext
from app.events import Emitter
from app.llm.router import Router
from app.rag import ScoredChunk


def make_settings(**over) -> Settings:
    base = dict(
        openai_api_key="sk-test-FAKEFAKEFAKEFAKE",
        openai_model="fake-openai-model",
        openrouter_api_key="",
        openrouter_base_url="http://invalid.local",
        gemini_model="fake-gemini",
        default_provider="openai",
        embedding_model="text-embedding-3-small",
        chroma_dir=Path("/nonexistent"),
        chroma_collection="none",
        top_k=6,
        max_supervisor_steps=6,
        max_tool_rounds=4,
        worker_timeout_s=10.0,
        request_budget_s=30.0,
        llm_timeout_s=10.0,
    )
    base.update(over)
    return Settings(**base)


def chunk(doc: str, sec: str, i: int, text: str, title: str = "Fake Paper", score: float = 0.8) -> ScoredChunk:
    cid = f"{doc}::{sec}::{i:03d}"
    md = {"doc_id": doc, "title": title, "section_path": sec, "html_anchor_url": f"https://arxiv.org/html/{doc}#{sec}",
          "published": "2024-01-01", "arxiv_id": doc.split("v")[0]}
    return ScoredChunk(cid, text, md, score)


class FakeRetriever:
    """search() returns chunks chosen by a function of the query; records calls."""

    def __init__(self, pick: Callable[[str], list[ScoredChunk]] | None = None, fail: Exception | None = None):
        self.pick = pick or (lambda q: [chunk("2401.00001v1", "S1", 0, "Alpha method retrieves when uncertain.")])
        self.fail = fail
        self.calls: list[tuple[str, int]] = []
        self.count = 3
        self.dims = 1536
        self.known: dict[str, ScoredChunk] = {}

    async def search(self, query: str, k: int):
        self.calls.append((query, k))
        await asyncio.sleep(0)
        if self.fail:
            raise self.fail
        out = self.pick(query)
        for c in out:
            self.known[c.chunk_id] = c
        return out

    def get_chunk(self, chunk_id: str):
        c = self.known.get(chunk_id)
        return None if c is None else {"chunk_id": chunk_id, "text": c.text, "doc_id": c.metadata["doc_id"]}


def tool_call(name: str, args: dict, i: int = 1) -> dict:
    return {"name": name, "args": args, "id": f"call_{i}", "type": "tool_call"}


@dataclass
class Script:
    """Scripted behaviour per call site. Each entry may be a value or an async callable(messages, kw)."""

    planner: Any = None  # dict plan args, AIMessage, Exception, or callable
    worker: list[Any] = field(default_factory=list)  # per-round responses (AIMessage / Exception / callable)
    synth_chunks: list[str] = field(default_factory=lambda: ["Answer ", "text."])
    synth_delay: float = 0.0
    synth_fail_after: int | None = None  # raise after N chunks
    calls: list[dict] = field(default_factory=list)
    synth_messages: list[BaseMessage] | None = None
    synth_completed: bool = False
    synth_cancelled: bool = False


class FakeChat(BaseChatModel):
    script: Any

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def bind_tools(self, tools, *, tool_choice=None, **kw):
        names = [getattr(t, "name", None) or getattr(t, "__name__", None) for t in tools]
        return self.bind(tool_names=names, tool_choice=tool_choice)

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        raise RuntimeError("sync path not expected")

    async def _resolve(self, item, messages, kw):
        if isinstance(item, Exception):
            raise item
        if callable(item):
            item = await item(messages, kw)
        return item

    async def _agenerate(self, messages, stop=None, run_manager=None, **kw):
        names = kw.get("tool_names") or []
        s: Script = self.script
        if "Plan" in names:
            s.calls.append({"site": "planner", "kw": kw})
            item = await self._resolve(s.planner, messages, kw)
            if isinstance(item, dict):
                item = AIMessage(content="", tool_calls=[tool_call("Plan", item)],
                                 usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
            return ChatResult(generations=[ChatGeneration(message=item)])
        if "corpus_search" in names:
            n = sum(1 for c in s.calls if c["site"] == "worker")
            s.calls.append({"site": "worker", "kw": kw, "tool_choice": kw.get("tool_choice"), "n_messages": len(messages)})
            item = s.worker[n] if n < len(s.worker) else AIMessage(content="no more script")
            item = await self._resolve(item, messages, kw)
            return ChatResult(generations=[ChatGeneration(message=item)])
        raise RuntimeError(f"unexpected non-streaming call with tools={names}")

    async def _astream(self, messages, stop=None, run_manager=None, **kw):
        s: Script = self.script
        s.calls.append({"site": "synth"})
        s.synth_messages = list(messages)
        try:
            for i, c in enumerate(s.synth_chunks):
                if s.synth_fail_after is not None and i >= s.synth_fail_after:
                    raise RuntimeError("stream broke: upstream reset sk-proj-SECRETSECRETSECRET1234")
                if s.synth_delay:
                    await asyncio.sleep(s.synth_delay)
                yield ChatGenerationChunk(message=AIMessageChunk(content=c))
            yield ChatGenerationChunk(message=AIMessageChunk(
                content="", usage_metadata={"input_tokens": 100, "output_tokens": len(s.synth_chunks), "total_tokens": 100 + len(s.synth_chunks)}))
            s.synth_completed = True
        except asyncio.CancelledError:
            s.synth_cancelled = True
            raise


class FakeRouter(Router):
    def __init__(self, settings: Settings, script: Script):
        super().__init__(settings)
        self.script = script

    def client(self, provider: str):
        return FakeChat(script=self.script)


def make_ctx(query: str, script: Script, retriever=None, request_id: str = "req", **settings_over) -> RequestContext:
    s = make_settings(**settings_over)
    return RequestContext(
        request_id=request_id, query=query, k=s.top_k, settings=s, router=FakeRouter(s, script),
        retriever=retriever or FakeRetriever(), emitter=Emitter(request_id),
    )


def drain(ctx: RequestContext) -> list[tuple[str, dict]]:
    out = []
    q = ctx.emitter.queue
    while not q.empty():
        item = q.get_nowait()
        if isinstance(item, tuple):
            out.append(item)
    return out


def plan(*tasks, summary="plan"):
    return {"plan_summary": summary, "tasks": [
        {"worker": w, "instruction": ins, "depends_on": list(deps)} for (w, ins, deps) in tasks]}
