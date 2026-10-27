"""Tester-owned Phase 6 retest-2 checks for the child-process SQL guard (P6-D1 fix in app/sqlguard.py).

Real SQLite against data/papers.db (read only); no providers. Covers: what environment the query child
receives, concurrent calls (isolation, cleanup, no leftover processes), the parent's kill-on-timeout path,
and how an unusable database is reported.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import app.sqlguard as sg
from app.config import ROOT
from app.sqlguard import SqlError, run_readonly

DB = ROOT / "data" / "papers.db"
CANARY = "sk-TESTCANARY-p6r2-0000000000000000"


def _truth(sql: str) -> list[list]:
    con = sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return [list(r) for r in con.execute(sql)]
    finally:
        con.close()


def _live_children() -> list[str]:
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,stat=,command="], capture_output=True, text=True).stdout
    me = str(os.getpid())
    return [ln for ln in out.splitlines() if ln.split()[1:2] == [me] and "sqlguard" in ln]


def test_T6_CHILD_ENV_no_credentials(monkeypatch):
    """The query child must not inherit provider/Tavily credentials or the app's environment."""
    for k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.setenv(k, CANARY)
    seen = {}
    real = subprocess.run

    def spy(*a, **kw):
        seen["env"] = dict(kw.get("env") or {})
        seen["argv"] = list(a[0])
        seen["input"] = kw.get("input", "")
        return real(*a, **kw)

    monkeypatch.setattr(sg.subprocess, "run", spy)
    res = run_readonly(DB, "SELECT count(*) FROM papers")
    assert res.rows == _truth("SELECT count(*) FROM papers")
    assert seen["env"], "child env must be explicit, not inherited"
    assert CANARY not in "".join(seen["env"].values()) + "".join(seen["argv"]) + seen["input"]
    assert set(seen["env"]) <= {"SQLITE_TMPDIR", "TMPDIR", "PATH"}, sorted(seen["env"])
    assert "-I" in seen["argv"]


QUERIES = [
    "SELECT count(*) FROM papers",
    "SELECT primary_category, count(*) FROM papers GROUP BY primary_category ORDER BY 2 DESC, 1",
    "SELECT arxiv_id, title FROM papers ORDER BY published DESC, arxiv_id LIMIT 7",
    "SELECT category, count(DISTINCT arxiv_id) FROM paper_categories GROUP BY category ORDER BY 1",
    "WITH y AS (SELECT year, count(*) c FROM papers GROUP BY year) SELECT * FROM y ORDER BY year",
    "SELECT count(*) FROM repos",
    "WITH RECURSIVE c(x) AS (SELECT 1 UNION SELECT x+1 FROM c WHERE x<20) SELECT sum(x) FROM c",
    "SELECT a.name, count(*) n FROM authors a JOIN paper_authors pa ON pa.author_id=a.author_id "
    "GROUP BY a.author_id ORDER BY n DESC, a.name LIMIT 5",
]


def test_T6_CHILD_CONCURRENT_isolated_and_clean(monkeypatch, tmp_path):
    """24 concurrent calls: each result equals its own independent truth; no temp dirs or children left."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    jobs = [QUERIES[i % len(QUERIES)] for i in range(24)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=12) as ex:
        results = list(ex.map(lambda q: (q, run_readonly(DB, q)), jobs))
    wall = time.perf_counter() - t0
    for q, r in results:
        assert r.rows == _truth(q)[:50], q
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("sqlguard-")]
    assert not _live_children()
    assert wall < 20, wall


def test_T6_CHILD_WALL_KILL_cleans_up(monkeypatch, tmp_path):
    """When the parent's wall cap fires, the result is sql_timeout, the child is gone and its temp dir removed."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(sg, "CHILD_WALL_S", 0.6)
    sql = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r) SELECT max(n) FROM (SELECT n FROM r LIMIT 4000000000)"
    t0 = time.perf_counter()
    with pytest.raises(SqlError) as ei:
        run_readonly(DB, sql, max_ops=10**12, max_seconds=60)  # in-child caps disabled so the parent cap fires
    assert ei.value.kind == "sql_timeout"
    assert time.perf_counter() - t0 < 3
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("sqlguard-")]
    assert not _live_children()


@pytest.mark.xfail(strict=True, reason="P6-D3: an unusable DB is reported as a resource-limit refusal")
def test_T6_CHILD_MISSING_DB_is_not_a_resource_refusal(tmp_path):
    """A missing metadata DB is an infrastructure failure, not 'simplify your query'."""
    with pytest.raises(SqlError) as ei:
        run_readonly(tmp_path / "missing.db", "SELECT count(*) FROM papers")
    msg = str(ei.value).lower()
    assert "resource limit" not in msg and "simplify" not in msg, (ei.value.kind, msg)
