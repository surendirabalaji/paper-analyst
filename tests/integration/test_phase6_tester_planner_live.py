"""Tester Phase 6 planner probe (REAL Gemini-via-OpenRouter planner call per question; RUN_LIVE=1).

Held-out phrasings, not taken from the developer's tests or prompts. Asserts only the worker class:
metadata-only questions must plan a data task first; content-only questions must not use data; mixed
questions must have data first and a research/summary task depending on it. Writes the raw plans.
Cost: about $0.002 per question.
"""
import json
import os
import pathlib

import pytest

pytestmark = [pytest.mark.live, pytest.mark.anyio]
OUT = pathlib.Path(__file__).resolve().parents[2] / "eval/results/testing" / os.environ.get("TESTER_RUN", "phase6-local")

CASES = [
    ("What fraction of the indexed papers have an accompanying code release?", "data"),
    ("List every paper in the collection whose primary arXiv category is cs.IR.", "data"),
    ("Which author appears on the most papers here?", "data"),
    ("When was the oldest paper in the collection first posted to arXiv?", "data"),
    ("Break down the corpus by topic group, with counts.", "data"),
    ("Did anyone named Wang co-author more than one indexed paper?", "data"),
    ("What is the main limitation of tree-of-thoughts search as discussed in the papers?", "content"),
    ("Explain the retrieval critique tokens used in Self-RAG.", "content"),
    ("How do the papers evaluate multi-agent debate against single-agent baselines?", "content"),
    ("Grab the three newest papers tagged cs.MA and tell me what each one contributes.", "mixed"),
    ("Among the papers that ship code, which ones from 2023 tackle tool use, and how do they work?", "mixed"),
    ("Give a quick summary of whatever the earliest 2024 paper in the index is.", "mixed"),
]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.parametrize("q,want", CASES)
async def test_T6_PLAN_live(q, want):
    from app.config import load_settings
    from app.context import RequestContext
    from app.events import Emitter
    from app.graph import planner
    from app.llm.router import Router

    s = load_settings()
    if not (s.openai_api_key and s.openrouter_api_key):
        pytest.skip("provider keys missing")
    ctx = RequestContext(request_id="t6plan", query=q, k=s.top_k, settings=s, router=Router(s),
                         retriever=None, emitter=Emitter("t6plan"), tavily=None)
    out = await planner({"query": q}, {"configurable": {"ctx": ctx}})
    tasks = out["tasks"]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "planner_probe.jsonl").open("a") as f:
        f.write(json.dumps({"q": q, "want": want, "plan_source": out.get("plan_source"),
                            "tasks": [(t["worker"], t["depends_on"], t["needs_live"], t["instruction"]) for t in tasks]}) + "\n")
    workers = [t["worker"] for t in tasks]
    assert out["plan_source"] == "llm"
    if want == "data":
        assert workers[0] == "data", workers
    elif want == "content":
        assert "data" not in workers, workers
    else:
        assert workers[0] == "data", workers
        assert any(t["worker"] in ("research", "summary") and tasks[0]["id"] in t["depends_on"] for t in tasks[1:]), tasks
