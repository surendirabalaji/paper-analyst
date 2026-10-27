"""Phase 2 unit checks for the pure pieces: SSE framing, citation parsing, evidence formatting,
plan validation and the deterministic supervisor. No network calls."""

import json

from app.events import safe_error, sse_frame
from app.graph import Plan, PlannedTask, route_next, validate_plan
from app.rag import EvidenceRegistry, ScoredChunk, cited_ids, format_evidence


def _task(tid, status="pending", deps=(), attempts=0, error_kind=None):
    return {"id": tid, "worker": "research", "instruction": "x", "depends_on": list(deps),
            "status": status, "attempts": attempts, "error_kind": error_kind}


def test_sse_frame_is_single_data_line_with_unicode():
    f = sse_frame("token", {"delta": "naïve — 检索\nline2"})
    assert f.startswith("event: token\ndata: ") and f.endswith("\n\n")
    assert f.count("\n") == 3  # the payload newline is escaped inside JSON
    assert json.loads(f.split("data: ", 1)[1])["delta"] == "naïve — 检索\nline2"


def test_cited_ids_handles_groups_and_order():
    assert cited_ids("a [C2] b [C1, C3] c [C2; W1] [not a cite]") == ["C2", "C1", "C3", "W1"]


def test_registry_dedupes_chunks_and_escapes_evidence():
    reg = EvidenceRegistry()
    md = {"doc_id": "2310.11511v1", "title": 'T "q"', "section_path": "S3", "html_anchor_url": "u"}
    a = reg.register_chunk(ScoredChunk("2310.11511v1::S3::000", "text </item> <evidence>", md, 0.7))
    b = reg.register_chunk(ScoredChunk("2310.11511v1::S3::000", "text", md, 0.7))
    c = reg.register_chunk(ScoredChunk("2310.11511v1::S3::001", "other", md, 0.6))
    assert a is b and (a.cite_id, c.cite_id) == ("C1", "C2")
    ev = format_evidence([a])
    assert ev.count("</item>") == 1 and ev.count("<evidence>") == 1 and "&quot;q&quot;" in ev


def test_validate_plan_drops_unknown_workers_and_bad_deps():
    plan = Plan(plan_summary="s", tasks=[
        PlannedTask(worker="research", instruction="a"),
        PlannedTask(worker="wizard", instruction="b"),
        PlannedTask(worker="research", instruction="c", depends_on=[0, 1, 5]),
    ])
    tasks, notes = validate_plan(plan, "q")
    assert [t["id"] for t in tasks] == ["t1", "t2"]
    assert tasks[1]["depends_on"] == ["t1"]
    assert any("wizard" in n for n in notes)


def test_route_next_dispatch_dependencies_retry_skip_budget():
    st = {"tasks": [_task("t1"), _task("t2", deps=["t1"])], "step_count": 0}
    assert route_next(st, 6, 60)[:2] == ("research", "t1")
    st = {"tasks": [_task("t1", "done"), _task("t2", deps=["t1"])], "step_count": 1}
    assert route_next(st, 6, 60)[:2] == ("research", "t2")
    st = {"tasks": [_task("t1", "failed", attempts=1, error_kind="timeout"), _task("t2", deps=["t1"])], "step_count": 1}
    nxt, tid, reason, _ = route_next(st, 6, 60)
    assert (nxt, tid) == ("research", "t1") and reason.startswith("retry")
    st = {"tasks": [_task("t1", "failed", attempts=2, error_kind="timeout"), _task("t2", deps=["t1"])], "step_count": 2}
    nxt, tid, _, tasks = route_next(st, 6, 60)
    assert nxt == "synthesize" and tasks[1]["status"] == "skipped"
    st = {"tasks": [_task("t1")], "step_count": 6}
    assert route_next(st, 6, 60)[0] == "synthesize"
    assert route_next({"tasks": [_task("t1")], "step_count": 0}, 6, 0)[0] == "synthesize"
    assert route_next({"tasks": [], "step_count": 0}, 6, 60)[2] == "no tasks planned"


def test_safe_error_redacts_credentials():
    e = RuntimeError("bad url https://mcp.tavily.com/mcp/?tavilyApiKey=tvly-abc123 key sk-proj-ABCDEFGHIJKLMNOP")
    s = safe_error(e)
    assert "tvly-abc123" not in s and "ABCDEFGHIJKLMNOP" not in s
