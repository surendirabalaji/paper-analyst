"""Tester Phase 6 repair-round checks with a REAL model (Gemini/OpenAI per policy), REAL papers.db; RUN_LIVE=1.

Fault injection is local only (monkeypatch of app.agents.data.run_readonly): the injected failures are real
SQLite errors produced by running a genuinely invalid query through the guard.
  R1: first query fails (unknown column) -> one repair -> correct count vs independent SQL truth; the repair
      call carries the repair_round feature and is routed to the capability side.
  R2: every query fails -> exactly MAX_SQL_FAILURES guard executions, task failed/invalid_sql, no D# cited,
      answer does not state a number, DB unchanged.
Cost: about $0.02-0.05 total.
"""
import hashlib
import json
import os
import pathlib
import re
import sqlite3
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from phase2_fakes import drain  # noqa: E402

pytestmark = [pytest.mark.live, pytest.mark.anyio]
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "eval/results/testing" / os.environ.get("TESTER_RUN", "phase6-local")
DB = ROOT / "data" / "papers.db"


@pytest.fixture
def anyio_backend():
    return "asyncio"


def ctx_for(q, rid):
    from app.config import load_settings
    from app.context import RequestContext
    from app.events import Emitter
    from app.llm.router import Router
    from app.rag import Retriever

    s = load_settings()
    if not (s.openai_api_key and s.openrouter_api_key):
        pytest.skip("provider keys missing")
    return RequestContext(request_id=rid, query=q, k=s.top_k, settings=s, router=Router(s),
                          retriever=Retriever(s), emitter=Emitter(rid), tavily=None)


def save(name, events):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text("".join(json.dumps({"event": e, "data": d}, default=str) + "\n" for e, d in events))


async def test_T6_R1_repair_then_correct_answer(monkeypatch):
    import app.agents.data as data
    from app.graph import run_request
    from app.sqlguard import run_readonly

    n = {"i": 0}

    def first_fails(db, sql, **kw):
        n["i"] += 1
        if n["i"] == 1:
            return run_readonly(db, "SELECT p.arxiv_id FROM papers p WHERE p.num_citations > 10")
        return run_readonly(db, sql, **kw)

    monkeypatch.setattr(data, "run_readonly", first_fails)
    before = hashlib.sha256(DB.read_bytes()).hexdigest()
    truth = sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True).execute(
        "SELECT count(DISTINCT arxiv_id) FROM paper_categories WHERE category = 'cs.AI'").fetchone()[0]
    ctx = ctx_for("How many indexed papers carry the cs.AI category at all (primary or cross-listed)?", "t6-r1")
    final = await run_request(ctx)
    events = drain(ctx)
    save("live_R1_repair.events.jsonl", events)
    sql = [d for e, d in events if e == "sql_result"]
    assert sql[0]["status"] == "error" and sql[0]["error_kind"] == "sql_error" and sql[0]["repair_rounds_left"] == 1
    assert any(d["status"] == "ok" for d in sql[1:])
    rep = [d for e, d in events if e == "llm_route" and d["agent"] == "data" and d["features"].get("repair_round")]
    assert rep and rep[0]["provider"] == "openai" and rep[0]["dimension"] == "capability" and rep[0]["rationale"]
    assert str(truth) in final["final_answer"], (truth, final["final_answer"][:300])
    assert hashlib.sha256(DB.read_bytes()).hexdigest() == before


async def test_T6_R2_all_queries_fail_is_an_honest_failure(monkeypatch):
    import app.agents.data as data
    from app.graph import run_request
    from app.sqlguard import run_readonly

    n = {"i": 0}

    def always_fails(db, sql, **kw):
        n["i"] += 1
        return run_readonly(db, "SELECT nope_col FROM papers")

    monkeypatch.setattr(data, "run_readonly", always_fails)
    ctx = ctx_for("How many papers in the collection were posted in 2025?", "t6-r2")
    final = await run_request(ctx)
    events = drain(ctx)
    save("live_R2_allfail.events.jsonl", events)
    assert n["i"] == data.MAX_SQL_FAILURES
    r = final["worker_results"]["t1"]
    assert r["status"] == "failed" and r["error_kind"] == "invalid_sql"
    assert not [d for e, d in events if e == "citation" and d["cite_id"].startswith("D")]
    done = [d for e, d in events if e == "done"][0]
    assert done["status"] in ("failed", "partial")
    assert not re.search(r"\b19\b", final["final_answer"] or ""), final["final_answer"]
