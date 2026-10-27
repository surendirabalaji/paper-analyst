"""Tester probe for P6-D1 retest 2: run run_readonly cases with TMPDIR on a RAM disk, sample peak disk use,
child peak RSS (RUSAGE_CHILDREN delta is a max, so run each case in a fresh parent), elapsed, outcome."""
import json, os, resource, shutil, subprocess, sys, threading, time

ROOT, DB, RAM = sys.argv[1], sys.argv[2], sys.argv[3]
R4 = "replace(replace(replace(replace('x','x','xxxxxxxxxx'),'x','xxxxxxxxxx'),'x','xxxxxxxxxx'),'x','xxxxxxxxxx')"
BIG = f"replace({R4},'x','xxxxxxxxx')"  # 90,000 chars
S10K = R4  # 10,000 chars

def derived(i, nrows, w):
    return f"(SELECT s{i} FROM (SELECT replace({w},'x','x')||n AS s{i} FROM r WHERE n<={nrows}) GROUP BY s{i}) t{i}"

CASES = {
    "wide400": f"WITH a(s) AS (SELECT {BIG}) SELECT " + ",".join(["s"] * 400) + " FROM a, papers",
    "wide32x90k": f"WITH a(s) AS (SELECT {BIG}) SELECT " + ",".join(f"s c{i}" for i in range(32)) + " FROM a, papers",
    "wide30x10k": f"WITH a(s) AS (SELECT {S10K}) SELECT " + ",".join(f"s c{i}" for i in range(30)) + " FROM a, papers",
    "spill_groupby": f"WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r WHERE n<1000000) SELECT s FROM (SELECT replace({R4},'x','xxxxx')||n AS s FROM r) GROUP BY s ORDER BY s DESC",
    "spill_10MB": f"WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r WHERE n<1000) SELECT s FROM (SELECT {S10K}||n AS s FROM r) GROUP BY s ORDER BY s DESC",
    "spill_30MB": f"WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r WHERE n<3000) SELECT s FROM (SELECT {S10K}||n AS s FROM r) GROUP BY s ORDER BY s DESC",
    "distinct_spill": f"WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r WHERE n<1000000) SELECT DISTINCT {R4}||n FROM r",
    "doubling": "WITH RECURSIVE d(s,n) AS (SELECT 'x',0 UNION ALL SELECT s||s,n+1 FROM d WHERE n<25) SELECT length(s) FROM d",
    # several materialized derived tables, each under the 16 MB per-file cap
    "multi_ephemeral": "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r WHERE n<1400) SELECT 1 FROM "
        + ", ".join(derived(i, 1400, "'" + "x" * 10 + "'||" + S10K) for i in range(6)) + " LIMIT 1",
    "ok_join": "SELECT p.arxiv_id, group_concat(a.name, '; ') FROM papers p JOIN paper_authors pa ON pa.arxiv_id=p.arxiv_id JOIN authors a ON a.author_id=pa.author_id GROUP BY p.arxiv_id ORDER BY p.published DESC LIMIT 5",
    "ok_star": "SELECT * FROM papers ORDER BY published LIMIT 3",
    "ok_union_cte": "WITH RECURSIVE c(x) AS (SELECT 1 UNION SELECT x+1 FROM c WHERE x<10) SELECT sum(x) FROM c",
}


for _k in (1, 2, 4, 6, 8):
    CASES[f"inline_{_k}"] = ("WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r WHERE n<1400) SELECT 1 FROM "
        + ", ".join(derived(i, 1400, "'" + "x" * 10 + "'||" + S10K) for i in range(_k)) + " LIMIT 1")
for _n in list(CASES):
    if not _n.startswith("inline"): del CASES[_n]

def main(name):
    sys.path.insert(0, ROOT)
    from pathlib import Path
    from app.sqlguard import SqlError, run_readonly
    sql = CASES[name]
    base = shutil.disk_usage(RAM).used
    peak = {"v": 0}; stop = threading.Event()
    def sample():
        while not stop.is_set():
            peak["v"] = max(peak["v"], shutil.disk_usage(RAM).used - base); time.sleep(0.002)
    th = threading.Thread(target=sample); th.start()
    t0 = time.perf_counter()
    try:
        r = run_readonly(Path(DB), sql)
        out = {"ok": True, "rows": len(r.rows), "cols": len(r.columns), "max_cell": max((len(str(c)) for row in r.rows for c in row), default=0), "first": str(r.rows[0][:2])[:80] if r.rows else None}
    except SqlError as e:
        out = {"ok": False, "kind": e.kind, "msg": str(e)[:160]}
    el = time.perf_counter() - t0
    stop.set(); th.join()
    ch = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss  # bytes on macOS
    me = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    left = [d for d in os.listdir(RAM) if d.startswith("sqlguard-")]
    out.update(case=name, sql_chars=len(sql), elapsed_s=round(el, 3), child_peak_rss_mb=round(ch / 1e6, 1),
               parent_peak_rss_mb=round(me / 1e6, 1), peak_tmp_mb=round(peak["v"] / 1e6, 1), leftover_tmpdirs=len(left))
    print(json.dumps(out))

if __name__ == "__main__":
    if len(sys.argv) > 4:
        main(sys.argv[4])
    else:
        for n in CASES:
            env = dict(os.environ, TMPDIR=RAM + "/")
            p = subprocess.run([sys.executable, __file__, ROOT, DB, RAM, n], capture_output=True, text=True, env=env, timeout=60)
            print(p.stdout.strip() or ("ERR " + n + " " + p.stderr[-300:]))
