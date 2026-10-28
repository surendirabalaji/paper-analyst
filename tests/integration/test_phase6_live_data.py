"""Phase 6 live checks (REAL OpenAI and Gemini-via-OpenRouter calls, real local index and papers.db; RUN_LIVE=1).

1. Data -> Summary hand-off through the full graph: Data runs read-only SQL, its paper ids reach Summary
   through depends_on (a `handoff` event), Summary reads those papers, and the answer cites D# and C#.
2. Repair round with a real model: the first run_sql call is made to fail with a real SQLite
   "no such column" error (fault injection at the guard), the model repairs the query, and the repair
   call is routed to the capability side by the `repair_round` feature.
3. The real planner sends metadata questions to Data and content questions to Research/Summary.
The DB file must be byte-identical afterwards. Cost: about $0.07-0.10 per run.
"""

import hashlib
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from phase2_fakes import drain  # noqa: E402

pytestmark = [pytest.mark.live, pytest.mark.anyio, pytest.mark.usefixtures("close_openai_clients")]  # P8b-D1
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "eval" / "results" / "phase6"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
def settings():
    from app.config import load_settings

    s = load_settings()
    if not (s.openai_api_key and s.openrouter_api_key):
        pytest.skip("OPENAI_API_KEY and OPENROUTER_API_KEY are required")
    return s


def make_ctx(settings, q, rid):
    from app.context import RequestContext
    from app.events import Emitter
    from app.llm.router import Router
    from app.rag import Retriever

    return RequestContext(request_id=rid, query=q, k=settings.top_k, settings=settings, router=Router(settings),
                          retriever=Retriever(settings), emitter=Emitter(rid), tavily=None)


def save(name, events):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text("".join(json.dumps({"event": e, "data": d}, default=str) + "\n" for e, d in events))


def db_hash(settings):
    return hashlib.sha256(pathlib.Path(settings.sqlite_path).read_bytes()).hexdigest()


async def test_data_to_summary_handoff_live(settings):
    from app.graph import run_request

    before = db_hash(settings)
    q = "Which multi-agent-systems papers from 2025 or 2026 are indexed? Summarize the two most recent."
    ctx = make_ctx(settings, q, "live6-handoff")
    final = await run_request(ctx)
    events = drain(ctx)
    save("live_test_handoff.events.jsonl", events)
    ev = lambda n: [d for e, d in events if e == n]  # noqa: E731

    assert [t["worker"] for t in final["tasks"]][:2] == ["data", "summary"]
    sql = [d for d in ev("sql_result") if d["status"] == "ok"]
    assert sql and sql[0]["n_rows"] >= 2 and all(c.startswith("D") for c in sql[0]["cite_ids"])
    (h,) = ev("handoff")
    handed = [p["arxiv_id"] for p in h["papers"]]
    assert h["to_agent"] == "summary" and len(handed) >= 2
    summary_papers = {c["args_safe"].get("paper") for c in ev("tool_call") if c["agent"] == "summary"}
    assert summary_papers & set(handed[:2]), (summary_papers, handed)
    cites = ev("citation")
    assert any(c["type"] == "sql_row" and c["sql"] and c["href"].startswith("https://arxiv.org/abs/") for c in cites)
    assert any(c["type"] == "paper_chunk" for c in cites)
    done = ev("done")[0]
    assert done["status"] in ("done", "partial") and not done["unresolved_citations"]
    assert db_hash(settings) == before


async def test_repair_round_live_with_injected_sql_error(settings, monkeypatch):
    import app.agents.data as data
    from app.graph import run_request
    from app.sqlguard import run_readonly

    calls = {"n": 0}

    def first_fails(db, sql, **kw):
        calls["n"] += 1
        if calls["n"] == 1:  # a real SQLite error message for a column that does not exist
            return run_readonly(db, "SELECT venue FROM papers")
        return run_readonly(db, sql, **kw)

    monkeypatch.setattr(data, "run_readonly", first_fails)
    ctx = make_ctx(settings, "How many papers in the corpus were published in 2024?", "live6-repair")
    final = await run_request(ctx)
    events = drain(ctx)
    save("live_test_repair.events.jsonl", events)
    sql = [d for e, d in events if e == "sql_result"]
    assert sql[0]["status"] == "error" and sql[0]["error_kind"] == "sql_error"
    assert any(d["status"] == "ok" for d in sql[1:]), sql
    routes = [d for e, d in events if e == "llm_route" and d["agent"] == "data"]
    repair = [d for d in routes if d["features"].get("repair_round")]
    assert repair and repair[0]["provider"] == "openai" and repair[0]["dimension"] == "capability"
    assert final["worker_results"]["t1"]["status"] in ("done", "partial")
    assert "14" in final["final_answer"]  # 14 papers from 2024 (see data/papers.db)


@pytest.mark.parametrize("q,want", [
    ("How many papers in the corpus have a linked GitHub repository?", "data"),
    ("Who are the authors of the ReAct paper?", "data"),
    ("How does Self-RAG decide when to retrieve?", "content"),
    ("Compare how Reflexion and Self-Refine use feedback.", "content"),
])
async def test_planner_routes_metadata_to_data_live(settings, q, want):
    from app.graph import planner

    ctx = make_ctx(settings, q, "live6-plan")
    out = await planner({"query": q}, {"configurable": {"ctx": ctx}})
    workers = [t["worker"] for t in out["tasks"]]
    assert out["plan_source"] == "llm"
    if want == "data":
        assert workers[0] == "data", workers
    else:
        assert "data" not in workers and set(workers) <= {"research", "summary"}, workers
