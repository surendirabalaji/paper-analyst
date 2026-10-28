"""Read-only, bounded SQL execution over the corpus metadata DB (`data/papers.db`) for the Data agent.

Layers, each enough on its own to stop a write; together they also bound cost and output size:
  1. Lexical check (`check_sql`): one SELECT/WITH statement, no `;` except one trailing, no `--` or `/* */`
     comments and no write/DDL/PRAGMA keywords outside string literals, balanced parentheses (so <q> cannot close the wrapper in step 4),
     at most MAX_SQL_CHARS characters.
  2. Connection opened with a `mode=ro` URI and `PRAGMA query_only=ON`.
  3. `set_authorizer`: only SQLITE_SELECT, SQLITE_READ (on the five corpus tables), SQLITE_FUNCTION
     (allowlisted functions) and SQLITE_RECURSIVE are permitted. PRAGMA, ATTACH, writes, DDL,
     sqlite_master and pragma table-valued functions are denied by SQLite itself before the statement runs.
  4. The query runs wrapped as `SELECT * FROM (\n<q>\n) LIMIT 51` (50 rows shown, a 51st flags truncation;
     the newlines mean nothing in <q> can comment out the wrapper), with `set_progress_handler` as an
     operation cap plus a wall-clock cap.
  5. Memory and temp storage (tester P6-D1): SQLite limits cap any single value (SQLITE_LIMIT_LENGTH) and the
     column count (SQLITE_LIMIT_COLUMN); rows are fetched one at a time against a byte budget and cells cut to
     MAX_CELL_CHARS. The query runs in a short-lived child process (about 15 ms to start) with
     RLIMIT_AS / RLIMIT_FSIZE / RLIMIT_CPU and a private SQLITE_TMPDIR, so a large sort's temp files
     (in-memory on Cloud Run) and the process's memory are bounded by the OS, and the parent kills it after
     CHILD_WALL_S. Only check_sql runs in the app process.

  6. In the child only, `PRAGMA temp_store=MEMORY` plus `PRAGMA hard_heap_limit` (CHILD_HEAP_BYTES): sort and
     materialisation temp tables live in SQLite's heap, so their total per query is capped, not just each
     file (tester P6-D1c).

Every failure raises SqlError with a `kind`. Problems with the query: `sql_rejected` (policy refusals and
resource limits), `sql_error` (invalid SQL such as an unknown column), `sql_timeout` (the op/time cap); the
Data agent gives the model one repair round for these. Problems with the service, not the query (P6-D3):
`sql_unavailable` (the database cannot be opened) and `sql_internal` (the child process crashed; its exit
code and stderr tail are logged); these are not repairable and do not use the repair round.
"""

from __future__ import annotations

import json
import logging
import os
import re
import resource
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

MAX_ROWS = 50
MAX_SQL_CHARS = 2000
MAX_CELL_CHARS = 300
PROGRESS_EVERY = 1000  # VM instructions between progress-handler calls
MAX_OPS = 5_000_000  # about 5M VM instructions (0.1-0.5 s on this DB)
MAX_SECONDS = 2.0
MAX_VALUE_BYTES = 100_000  # largest string/blob any expression may build (SQLITE_LIMIT_LENGTH); cells are cut at 300
MAX_COLUMNS = 32  # SQLITE_LIMIT_COLUMN
MAX_RESULT_BYTES = 1_000_000  # raw bytes fetched (before cells are cut) across all returned rows
# The query child process (tester P6-D1 reopened): limits on the process that runs the query.
CHILD_MEM_BYTES = 512 * 1024 * 1024  # RLIMIT_AS (address space; Linux enforces it)
CHILD_FILE_BYTES = 16 * 1024 * 1024  # RLIMIT_FSIZE: largest file it may write (SQLite temp files)
CHILD_CPU_S = 5  # RLIMIT_CPU
CHILD_WALL_S = 8.0  # the parent kills it after this
CHILD_HEAP_BYTES = 32 * 1024 * 1024  # PRAGMA hard_heap_limit: all SQLite memory, incl. in-memory temp tables
INFRA_KINDS = frozenset({"sql_unavailable", "sql_internal"})  # service problems, not the model's query

log = logging.getLogger("paper_analyst.sqlguard")

ALLOWED_TABLES = frozenset({"papers", "authors", "paper_authors", "paper_categories", "repos"})
# printf/format are left out: one call can pad a value to any width (tester P6-D1), and over the length
# limit they return NULL silently instead of failing.
ALLOWED_FUNCTIONS = frozenset({
    # aggregates
    "count", "sum", "total", "avg", "min", "max", "group_concat", "string_agg",
    # scalar
    "abs", "coalesce", "ifnull", "nullif", "iif", "instr", "length", "lower", "upper", "ltrim", "rtrim",
    "trim", "replace", "substr", "substring", "round", "like", "glob", "cast",
    "date", "time", "datetime", "julianday", "strftime", "unixepoch", "typeof",
    # window
    "row_number", "rank", "dense_rank", "lag", "lead", "first_value", "last_value",
})
# SQLITE_RECURSIVE (WITH RECURSIVE) is read-only; the op cap bounds it.
_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
_START = re.compile(r"^\s*(select|with)\b", re.I)
# Statement keywords that never belong in a read query. Checked outside literals and quoted identifiers,
# so a title containing "update" is fine; the authorizer would refuse them anyway, but naming the refusal
# keeps the model from "repairing" a write into another write.
_WRITE_WORDS = re.compile(r"\b(insert|update|delete|drop|create|alter|attach|detach|pragma|vacuum|reindex|"
                          r"analyze|begin|commit|rollback|savepoint|release)\b", re.I)


class SqlError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass
class SqlResult:
    sql: str  # the model's query as checked (trailing ';' removed)
    executed_sql: str  # the wrapped statement actually run
    columns: list[str]
    rows: list[list]
    truncated: bool  # hit MAX_ROWS
    elapsed_ms: int
    denied: list[str] = field(default_factory=list)


def check_sql(sql: str) -> str:
    """Lexical policy. Returns the statement without a trailing ';' or raises SqlError(sql_rejected)."""
    q = (sql or "").strip()
    if not q:
        raise SqlError("sql_rejected", "empty query")
    if len(q) > MAX_SQL_CHARS:
        raise SqlError("sql_rejected", f"query longer than {MAX_SQL_CHARS} characters")
    if "\x00" in q:
        raise SqlError("sql_rejected", "NUL byte in query")
    quote = None
    depth = 0
    bare: list[str] = []  # the query text outside literals and quoted identifiers
    i = 0
    while i < len(q):
        ch = q[i]
        if not quote and ch not in ("'", '"', "`", "["):
            bare.append(ch)
        if quote:
            if ch == quote:
                if i + 1 < len(q) and q[i + 1] == quote:  # doubled quote inside a literal
                    i += 2
                    continue
                quote = None
        elif ch in ("'", '"', "`"):
            quote = ch
        elif ch == "[":
            quote = "]"
        elif ch == "-" and q[i:i + 2] == "--" or ch == "/" and q[i:i + 2] == "/*":
            raise SqlError("sql_rejected", "SQL comments are not allowed")
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:  # would close the LIMIT wrapper early
                raise SqlError("sql_rejected", "unbalanced parentheses")
        elif ch == ";":
            if q[i + 1:].strip():
                raise SqlError("sql_rejected", "only one statement is allowed")
            q = q[:i].rstrip()
            break
        i += 1
    if quote:
        raise SqlError("sql_error", "unterminated string or identifier")
    if depth:
        raise SqlError("sql_error", "unbalanced parentheses")
    if not _START.match(q):
        raise SqlError("sql_rejected", "only SELECT (or WITH ... SELECT) queries are allowed")
    word = _WRITE_WORDS.search("".join(bare))
    if word:
        raise SqlError("sql_rejected", f"{word.group(1).upper()} is not allowed: read-only SELECT queries only")
    return q


def wrap(sql: str, limit: int = MAX_ROWS) -> str:
    return f"SELECT * FROM (\n{sql}\n) LIMIT {int(limit)}"


def connect_ro(db_path: Path) -> sqlite3.Connection:
    """Read-only connection (URI mode=ro + query_only). Raises if the file is missing (mode=ro never creates)."""
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True, check_same_thread=False)
    con.execute("PRAGMA query_only = ON")
    # Memory bound (tester P6-D1): the op cap does not stop a few operations from building a huge value
    # (`s || s` doubling in a recursive CTE, `printf('%200000000s', …)`), so cap every value's size.
    con.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_VALUE_BYTES)
    con.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, MAX_SQL_CHARS + 100)
    con.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, MAX_COLUMNS)
    return con


def _cell(v):
    if isinstance(v, bytes):
        return f"<{len(v)} bytes>"
    if isinstance(v, str) and len(v) > MAX_CELL_CHARS:
        return v[:MAX_CELL_CHARS] + "…"
    return v


def run_readonly(db_path: Path, sql: str, *, max_rows: int = MAX_ROWS, max_ops: int = MAX_OPS,
                 max_seconds: float = MAX_SECONDS) -> SqlResult:
    """Check the query here, then run it in a short-lived child process under OS resource limits (blocking;
    call in a thread). The child is the memory/temp-file bound (tester P6-D1 reopened): SQLite limits cap
    single values, but many wide columns or a large sort's temp files (in-memory on Cloud Run) are only
    bounded by limits on the process that runs the query."""
    q = check_sql(sql)
    executed = wrap(q, max_rows + 1)  # one extra row tells us the result was cut
    job = {"db": str(Path(db_path).resolve()), "sql": executed, "max_rows": max_rows, "max_ops": max_ops,
           "max_seconds": max_seconds}
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="sqlguard-") as tmp:
        env = {"SQLITE_TMPDIR": tmp, "TMPDIR": tmp, "PATH": os.environ.get("PATH", "")}
        try:
            p = subprocess.run([sys.executable, "-I", "-S", "-c", _CHILD, str(_ROOT)], input=json.dumps(job),
                               capture_output=True, text=True, env=env, timeout=CHILD_WALL_S)
        except subprocess.TimeoutExpired:
            raise SqlError("sql_timeout", f"query stopped after {CHILD_WALL_S:g} s; simplify it") from None
    try:
        out = json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        if p.returncode < 0 or "MemoryError" in (p.stderr or ""):  # killed by a limit (CPU, memory) or OOM
            raise SqlError("sql_rejected", "query exceeded the resource limits (memory, CPU or temporary "
                                           "storage); simplify it") from None
        # Anything else is a crash of the query process itself (P6-D3): log it, and do not blame the query.
        from app.events import redact

        tail = redact((p.stderr or "").strip()[-600:])
        log.warning("sql child process failed: returncode=%s stderr_tail=%r", p.returncode, tail)
        raise SqlError("sql_internal", f"the query process failed unexpectedly (exit code {p.returncode}); "
                                       "this is a service problem, not a problem with the query") from None
    if "error" in out:
        raise SqlError(out["error"]["kind"], out["error"]["message"])
    return SqlResult(sql=q, executed_sql=executed, columns=out["columns"], rows=out["rows"],
                     truncated=out["truncated"], elapsed_ms=int((time.perf_counter() - t0) * 1000),
                     denied=out["denied"])


def _set_child_limits() -> None:
    """Runs in the child: cap address space (enforced on Linux, i.e. Cloud Run), each written file (SQLite
    temp files), CPU time, and open files. Over-size writes then fail with EFBIG instead of SIGXFSZ."""
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    for name, value in (("RLIMIT_AS", CHILD_MEM_BYTES), ("RLIMIT_FSIZE", CHILD_FILE_BYTES),
                        ("RLIMIT_CPU", CHILD_CPU_S), ("RLIMIT_NOFILE", 64)):
        res = getattr(resource, name)
        soft, hard = resource.getrlimit(res)
        lim = value if hard == resource.RLIM_INFINITY else min(value, hard)
        try:
            resource.setrlimit(res, (lim, hard))
        except (ValueError, OSError):  # e.g. RLIMIT_AS is not settable on macOS; Linux enforces it
            pass


def _child_main() -> None:
    """Entry point of the query child: read the job from stdin, print one JSON line."""
    _set_child_limits()
    job = json.loads(sys.stdin.read())
    try:
        out = _execute(Path(job["db"]), job["sql"], job["max_rows"], job["max_ops"], job["max_seconds"])
    except SqlError as e:
        out = {"error": {"kind": e.kind, "message": str(e)}}
    except MemoryError:
        out = {"error": {"kind": "sql_rejected", "message": "query needs too much memory; simplify it"}}
    sys.stdout.write(json.dumps(out) + "\n")


_ROOT = Path(__file__).resolve().parents[1]
_CHILD = "import sys; sys.path.insert(0, sys.argv[1]); from app.sqlguard import _child_main; _child_main()"


def _execute(db_path: Path, executed: str, max_rows: int, max_ops: int, max_seconds: float) -> dict:
    """Run the wrapped query under the authorizer, op/time caps and a result byte budget (in the child)."""
    denied: list[str] = []

    def authorizer(action, arg1, arg2, dbname, trigger):
        if action not in _ALLOWED_ACTIONS:
            denied.append(f"action {action}" + (f" ({arg1})" if arg1 else ""))
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_READ and (arg1 or "").lower() not in ALLOWED_TABLES:
            denied.append(f"read of {arg1!r}")
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() not in ALLOWED_FUNCTIONS:
            denied.append(f"function {arg2!r}")
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    ops = {"n": 0}
    deadline = time.perf_counter() + max_seconds

    def progress():
        ops["n"] += PROGRESS_EVERY
        return 1 if ops["n"] > max_ops or time.perf_counter() > deadline else 0

    try:
        con = connect_ro(db_path)
        # Child-only, process-wide: temp tables in memory, and all SQLite memory capped (P6-D1c).
        con.execute("PRAGMA temp_store = MEMORY")
        con.execute(f"PRAGMA hard_heap_limit = {int(CHILD_HEAP_BYTES)}")
        con.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchall()  # fails here if the file is not a database
    except sqlite3.DatabaseError as e:
        msg = str(e)
        if "not a database" in msg:  # a corrupt or wrong file: reported as before
            raise SqlError("sql_error", msg) from None
        raise SqlError("sql_unavailable", f"the corpus database cannot be opened ({msg}); this is a service "
                                          "problem, not a problem with the query") from None
    rows, used = [], 0
    try:
        con.set_authorizer(authorizer)
        con.set_progress_handler(progress, PROGRESS_EVERY)
        try:
            cur = con.execute(executed)
            columns = [d[0] for d in cur.description or []]
            # Row by row with a byte budget, so wide rows never pile up (tester P6-D1a).
            while len(rows) <= max_rows:
                row = cur.fetchone()
                if row is None:
                    break
                used += sum(len(v) if isinstance(v, (str, bytes)) else 8 for v in row)
                if used > MAX_RESULT_BYTES:
                    raise SqlError("sql_rejected", f"result over {MAX_RESULT_BYTES:,} bytes before truncation; "
                                                   "select fewer or shorter columns")
                rows.append([_cell(v) for v in row])
        except sqlite3.DatabaseError as e:
            msg = str(e)
            if "too big" in msg or "too many columns" in msg:  # SQLITE_LIMIT_LENGTH / SQLITE_LIMIT_COLUMN
                raise SqlError("sql_rejected", f"query builds a value over the {MAX_VALUE_BYTES:,}-byte limit or "
                                               f"more than {MAX_COLUMNS} columns; select fewer or shorter columns") from None
            if "disk is full" in msg or "I/O error" in msg or "unable to open" in msg or "out of memory" in msg:
                raise SqlError("sql_rejected", "query needs too much temporary storage or memory (large sort, "
                                               "GROUP BY or DISTINCT); simplify it") from None
            if "interrupted" in msg:
                raise SqlError("sql_timeout", f"query stopped: over the {max_ops:,}-operation / "
                                              f"{max_seconds:g} s cap; simplify it") from None
            if denied or "not authorized" in msg or "prohibited" in msg:
                raise SqlError("sql_rejected", f"not allowed ({'; '.join(denied[:3]) or msg}); "
                                               "only SELECT over the corpus tables is permitted") from None
            raise SqlError("sql_error", msg) from None
        except sqlite3.Warning as e:  # e.g. "You can only execute one statement at a time."
            raise SqlError("sql_rejected", str(e)) from None
    finally:
        con.close()
    return {"columns": columns, "rows": rows[:max_rows], "truncated": len(rows) > max_rows, "denied": denied}


def schema_text(db_path: Path) -> str:
    """DDL of the corpus tables plus value hints (trusted, fixed queries; not model SQL)."""
    con = connect_ro(db_path)
    try:
        ddl = [r[0] for r in con.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name IN "
            f"({','.join('?' * len(ALLOWED_TABLES))}) ORDER BY name", sorted(ALLOWED_TABLES))]
        y0, y1, n = con.execute("SELECT min(year), max(year), count(*) FROM papers").fetchone()
        d0, d1 = con.execute("SELECT min(published), max(published) FROM papers").fetchone()
        cats = [r[0] for r in con.execute(
            "SELECT category || ' (' || count(*) || ')' FROM paper_categories GROUP BY category ORDER BY count(*) DESC")]
        topics = [r[0] for r in con.execute("SELECT DISTINCT topic FROM papers ORDER BY topic")]
        n_repos = con.execute("SELECT count(*) FROM repos").fetchone()[0]
    finally:
        con.close()
    return "\n".join([
        *ddl,
        "",
        "Value hints:",
        f"- papers: {n} rows; year {y0}-{y1}; published (v1 submission date, YYYY-MM-DD) {d0} .. {d1}.",
        "- arxiv_id is the base id without version (e.g. 2310.11511); doc_id has the version (2310.11511v2).",
        "- A paper's categories are in paper_categories (all of them, is_primary=1 for the primary one);",
        "  papers.primary_category holds only the primary one. Category counts: " + ", ".join(cats),
        "- topic values: " + ", ".join(topics),
        f"- repos: {n_repos} papers have a linked GitHub repo (one row per paper; no row = no repo found).",
        "- Author names are as listed on arXiv; match them with LIKE '%Surname%'.",
        f"- Results are capped at {MAX_ROWS} rows; long text cells are cut at {MAX_CELL_CHARS} characters.",
    ])
