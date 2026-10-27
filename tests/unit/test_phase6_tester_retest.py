"""Tester-owned Phase 6 retest checks (P6-D1 reopened: memory bounds beyond the per-value length limit).

SQLITE_LIMIT_LENGTH caps each value, but (a) a single row can repeat a large value in many result columns
and fetchmany() materialises 51 such rows in Python before cells are cut to 300 characters, and (b) a
GROUP BY / sort over many large rows spills to SQLite temp files, which on Cloud Run live in the
in-memory filesystem and count against the instance memory limit. Real SQLite against data/papers.db (read
only); no providers. Each check is sized so that a failing run stays well below 1 GB.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import tracemalloc

import pytest

from app.config import ROOT
from app.sqlguard import SqlError, run_readonly

DB = ROOT / "data" / "papers.db"
R4 = "replace(replace(replace(replace('x','x','xxxxxxxxxx'),'x','xxxxxxxxxx'),'x','xxxxxxxxxx'),'x','xxxxxxxxxx')"
BIG = f"replace({R4},'x','xxxxxxxxx')"  # 90,000 characters, under the 100,000-byte value limit
PY_BUDGET = 32 * 1024 * 1024  # bytes of Python allocation one accepted query may cause


def test_T6_SQL_MEM_wide_rows_are_bounded():
    """40 columns x 90 KB x 51 rows would be ~180 MB in Python if unbounded (400 columns measured 2 GB RSS)."""
    cols = ", ".join(f"s AS c{i}" for i in range(40))
    sql = f"WITH a(s) AS (SELECT {BIG}) SELECT {cols} FROM a, papers"
    tracemalloc.start()
    try:
        try:
            run_readonly(DB, sql)
        except SqlError:
            pass  # a refusal is an acceptable bound
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < PY_BUDGET, f"accepted query allocated {peak / 1e6:.0f} MB in Python"


_SPILL = textwrap.dedent("""
    import json, resource, signal, sys
    sys.path.insert(0, sys.argv[1])
    from pathlib import Path
    from app.sqlguard import SqlError, run_readonly
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)  # over-size writes fail with EFBIG instead of killing us
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
    try:
        run_readonly(Path(sys.argv[2]), sys.argv[3])
        print(json.dumps({"ok": True}))
    except SqlError as e:
        print(json.dumps({"ok": False, "kind": e.kind, "msg": str(e)}))
""")


def test_T6_SQL_MEM_temp_spill_is_bounded():
    """A GROUP BY over generated 50 KB rows must not write 64 MB or more of SQLite temp files (measured >1 GB
    in 0.5 s before this check). Run in a subprocess with RLIMIT_FSIZE so a failure cannot fill the disk."""
    sql = (f"WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r WHERE n<1000000) "
           f"SELECT s FROM (SELECT replace({R4},'x','xxxxx')||n AS s FROM r) GROUP BY s ORDER BY s DESC")
    p = subprocess.run([sys.executable, "-I", "-c", _SPILL, str(ROOT), str(DB), sql],
                       capture_output=True, text=True, timeout=60)
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out.get("ok") or out["kind"] in ("sql_rejected", "sql_timeout"), out
    assert "full" not in out.get("msg", "") and "I/O" not in out.get("msg", ""), out
