"""Tester-owned Phase 6 checks (independent of the developer's test_phase6_data.py).

Covers: memory/size bounds of the SQL guard (not only rows/ops/time), accepted SELECT forms that must not
be rejected, independent SQL truth for the Data -> Summary hand-off, the no_upstream_papers rule when Data
returns rows without an arxiv_id column, failed-Data honesty, and concurrent-request isolation of the D#
registry. SQL runs for real against data/papers.db (which must stay byte-identical); providers are faked
(MOCKED evidence).
"""

from __future__ import annotations

import asyncio
import hashlib
import resource
import sqlite3

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.config import ROOT
from app.graph import NO_UPSTREAM_PAPERS, run_request
from app.sqlguard import SqlError, run_readonly
from tests.phase2_fakes import drain
from tests.unit.test_phase6_data import calls, ctx6, ev, script_for

DB = ROOT / "data" / "papers.db"
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _hash() -> str:
    return hashlib.sha256(DB.read_bytes()).hexdigest()


@pytest.fixture(scope="module", autouse=True)
def db_unchanged():
    before = _hash()
    yield
    assert _hash() == before, "papers.db changed during tester Phase 6 tests"


def truth(sql: str, *args):
    """Independent truth: plain read-only sqlite3, not app.sqlguard."""
    con = sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return con.execute(sql, args).fetchall()
    finally:
        con.close()


# ---------------------------------------------------------------- T6-SQL-MEM: value-size bounds
# A few VM operations can build a very large string, so the op/time caps do not bound memory.
# Sizes here are kept small (20-64 MB) so the test is safe to run; the defect scales to the 1e9-byte
# SQLite default (several GB of RSS) and can OOM a 2 GiB Cloud Run instance shared by 10 requests.

@pytest.mark.parametrize("sql", [
    "SELECT length(printf('%20000000s', 'x')) AS n",
    "SELECT length(format('%20000000s', 'x')) AS n",
    "WITH RECURSIVE r(s, n) AS (SELECT 'xxxxxxxx', 0 UNION ALL SELECT s || s, n + 1 FROM r WHERE n < 23) "
    "SELECT max(length(s)) AS n FROM r",
])
def test_T6_SQL_MEM_large_values_are_refused(sql):
    r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        res = run_readonly(DB, sql)
    except SqlError:
        return  # refused: the bound works
    built = res.rows[0][0]
    grew_mb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - r0) / 1e6  # macOS: bytes
    pytest.fail(f"guard built a {built:,}-character value (peak RSS grew ~{grew_mb:.0f} MB); "
                "no SQLITE_LIMIT_LENGTH-style cap")


# ---------------------------------------------------------------- T6-SQL-OK: safe forms are accepted

@pytest.mark.parametrize("sql", [
    "SELECT arxiv_id, updated FROM papers ORDER BY updated DESC",  # column named like a write word
    "SELECT count(*) AS n FROM papers WHERE title LIKE '%Delete%' OR title LIKE '%drop%'",
    "WITH t AS (SELECT arxiv_id FROM papers WHERE year = 2024) SELECT count(*) FROM t",
    "WITH RECURSIVE y(v) AS (SELECT 2022 UNION ALL SELECT v + 1 FROM y WHERE v < 2026) "
    "SELECT y.v, count(p.arxiv_id) FROM y LEFT JOIN papers p ON p.year = y.v GROUP BY y.v ORDER BY y.v",
    "SELECT arxiv_id, rank() OVER (PARTITION BY year ORDER BY published) AS r FROM papers ORDER BY year, r",
    "SELECT arxiv_id FROM papers WHERE year = 2023 UNION SELECT arxiv_id FROM repos ORDER BY 1",
    "SELECT count(*) FROM papers p WHERE EXISTS (SELECT 1 FROM repos r WHERE r.arxiv_id = p.arxiv_id)",
    "SELECT CASE WHEN year >= 2025 THEN 'new' ELSE 'old' END AS k, count(*) FROM papers GROUP BY k",
    "SELECT count(*) FROM authors WHERE name LIKE '%ü%' OR name LIKE '%李%'",
    "select count(*) from papers;",
])
def test_T6_SQL_OK_accepted_select_forms(sql):
    res = run_readonly(DB, sql)
    assert res.columns


def test_T6_SQL_OK_results_match_independent_sql():
    q = ("SELECT p.arxiv_id FROM papers p JOIN paper_categories c USING (arxiv_id) "
         "WHERE c.category = 'cs.CL' ORDER BY p.arxiv_id")
    assert [r[0] for r in run_readonly(DB, q).rows] == [r[0] for r in truth(q)][:50]
    n_auth = truth("SELECT count(DISTINCT author_id) FROM paper_authors")[0][0]
    assert run_readonly(DB, "SELECT count(DISTINCT author_id) FROM paper_authors").rows[0][0] == n_auth


# ---------------------------------------------------------------- T6-HO: hand-off uses Data's actual result

async def test_T6_HO_handoff_ids_equal_independent_truth():
    q = ("SELECT p.arxiv_id, p.title, p.published FROM papers p JOIN repos r USING (arxiv_id) "
         "WHERE p.year = 2022 ORDER BY p.published")
    expected = [r[0] for r in truth(q)]
    assert expected, "fixture assumption: some 2022 papers have repos"
    seen = {}

    async def summary_round(messages, kw):
        seen["user"] = next(m for m in messages if isinstance(m, HumanMessage)).content
        return AIMessage(content="- summary [D1].")

    n = len(expected)
    marks = " ".join(f"[D{i}]" for i in range(1, n + 1))
    s = script_for({"data": [calls(("run_sql", {"query": q})), AIMessage(content=f"{n} papers {marks}.")],
                    "summary": [summary_round]},
                   [("data", "2022 papers with code", []), ("summary", "summarise them", [0])])
    ctx = ctx6("Which 2022 papers have code? Summarise them.", s)
    final = await run_request(ctx)
    events = drain(ctx)
    (h,) = ev(events, "handoff")
    assert [p["arxiv_id"] for p in h["papers"]] == expected[:10]
    for aid in expected[:10]:
        assert f"arXiv:{aid}" in seen["user"]
    for p in h["papers"]:
        ref = ctx.registry.get(p["cite_id"])
        assert ref.type == "sql_row" and ref.href == f"https://arxiv.org/abs/{p['arxiv_id']}"
        assert ref.rows[0][0] == p["arxiv_id"] and ref.sql == q
    assert final["tasks"][1]["status"] == "done"


# ---------------------------------------------------------------- T6-NOPAPERS: rows without arxiv_id

async def test_T6_NOPAPERS_rows_without_arxiv_id_are_not_reported_as_no_matching_papers():
    """Data returned real rows (titles) but no arxiv_id column. The supervisor must not record that the
    lookup 'found no matching papers' (a false statement handed to the synthesizer)."""
    q = "SELECT title, published FROM papers WHERE year = 2025 ORDER BY published DESC LIMIT 2"
    assert len(truth(q)) == 2
    s = script_for({"data": [calls(("run_sql", {"query": q})), AIMessage(content="Two papers [D1].")],
                    "summary": [AIMessage(content="- summary.")]},
                   [("data", "two most recent 2025 papers", []), ("summary", "summarise them", [0])])
    ctx = ctx6("Summarise the two most recent 2025 papers in the corpus", s)
    final = await run_request(ctx)
    t2 = final["tasks"][1]
    synth_input = s.synth_messages[-1].content if s.synth_messages else ""
    assert not (t2["status"] == "skipped" and t2.get("error_kind") == NO_UPSTREAM_PAPERS), (
        "Summary skipped as no_upstream_papers although Data returned 2 rows; synthesizer was told: "
        + ("'found no matching papers'" if "found no matching papers" in synth_input else "(other)"))


# ---------------------------------------------------------------- T6-FAIL: failed Data is not success

async def test_T6_FAIL_rejected_only_queries_give_failed_data_and_no_d_citation():
    s = script_for({"data": [calls(("run_sql", {"query": "SELECT * FROM sqlite_master"})),
                             calls(("run_sql", {"query": "SELECT * FROM pragma_table_info('papers')"})),
                             AIMessage(content="There are 12 tables [D1].")]},
                   [("data", "list tables", [])], synth=("Unknown.",))
    ctx = ctx6("List the database tables", s)
    final = await run_request(ctx)
    events = drain(ctx)
    r = final["worker_results"]["t1"]
    assert r["status"] == "failed" and r["paper_ids"] == [] and r["sql_ok"] == 0
    assert ctx.registry.get("D1") is None
    assert "D1" in r["unknown_markers"] or "D1" not in r["evidence_ids"]
    assert not [c for c in ev(events, "citation") if c["cite_id"].startswith("D")]


# ---------------------------------------------------------------- T6-ISO: concurrent requests

async def test_T6_ISO_concurrent_data_requests_have_independent_registries():
    qa = "SELECT count(*) AS n FROM papers WHERE year = 2023"
    qb = "SELECT arxiv_id, title, published FROM papers WHERE year = 2024 ORDER BY published LIMIT 3"

    async def slow(item):
        async def f(messages, kw):
            await asyncio.sleep(0.05)
            return item
        return f

    sa = script_for({"data": [await slow(calls(("run_sql", {"query": qa}))), AIMessage(content="n [D1].")]},
                    [("data", "count 2023", [])], synth=("A ", "[D1]."))
    sb = script_for({"data": [calls(("run_sql", {"query": qb})), AIMessage(content="3 papers [D1] [D2] [D3].")]},
                    [("data", "list 2024", [])], synth=("B ", "[D1] [D2] [D3]."))
    ca, cb = ctx6("count 2023 papers", sa), ctx6("list 2024 papers", sb)
    ca.request_id, cb.request_id = "reqA", "reqB"
    await asyncio.gather(run_request(ca), run_request(cb))
    ea, eb = drain(ca), drain(cb)
    assert ca.registry.get("D1").type == "sql_result" and ca.registry.get("D2") is None
    assert [ca.registry.get("D1").sql] == [qa]
    assert {cb.registry.get(f"D{i}").type for i in (1, 2, 3)} == {"sql_row"}
    assert all(cb.registry.get(f"D{i}").sql == qb for i in (1, 2, 3))
    assert all(d["sql"] == qa for d in ev(ea, "sql_result")) and all(d["sql"] == qb for d in ev(eb, "sql_result"))
