"""Phase 7 live check (REAL OpenAI and Gemini-via-OpenRouter calls through the real FastAPI app, real local
index and papers.db; RUN_LIVE=1).

A two-part question (a metadata count for Data, a content question for Research or Summary, whichever the
planner picks) is sent through POST /api/chat with an injected failure of the content worker
(`worker:research:exception,worker:summary:exception`, X-Fault-Inject + X-Fault-Token). Expected: the request
completes with `partial`, the Data part is answered with D# citations, the answer's "Incomplete evidence"
note names the failed worker and labels the fault as injected, and the trace has the fault_injected event. The same header without the token is refused with 403. Cost: about $0.03-0.06.
"""

import json
import pathlib
import secrets

import pytest

pytestmark = [pytest.mark.live]
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "eval" / "results" / "phase7"
FAULT = "worker:research:exception,worker:summary:exception"
QUERY = "How many papers in the corpus have cs.MA as a category, and how does Self-RAG decide when to retrieve?"


def parse_sse(text):
    out = []
    for frame in text.split("\n\n"):
        ev, data = None, None
        for line in frame.splitlines():
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if ev:
            out.append((ev, data))
    return out


@pytest.fixture(scope="module")
def client():
    import os

    from fastapi.testclient import TestClient

    from app.config import load_settings

    s = load_settings()
    if not (s.openai_api_key and s.openrouter_api_key):
        pytest.skip("OPENAI_API_KEY and OPENROUTER_API_KEY are required")
    token = secrets.token_urlsafe(32)
    old = os.environ.get("FAULT_INJECT_TOKEN")
    os.environ["FAULT_INJECT_TOKEN"] = token
    from app.main import app

    with TestClient(app) as c:
        c.fault_token = token
        yield c
    if old is None:
        os.environ.pop("FAULT_INJECT_TOKEN", None)
    else:
        os.environ["FAULT_INJECT_TOKEN"] = old


def test_fault_header_without_token_is_refused(client):
    r = client.post("/api/chat", json={"query": QUERY}, headers={"X-Fault-Inject": "worker:research:exception"})
    assert r.status_code == 403


def test_live_injected_content_worker_failure_gives_an_honest_partial_answer(client):
    r = client.post("/api/chat", json={"query": QUERY},
                    headers={"X-Fault-Inject": FAULT, "X-Fault-Token": client.fault_token})
    assert r.status_code == 200
    events = parse_sse(r.text)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "live_injected_research_failure.events.jsonl").write_text(
        "".join(json.dumps({"event": e, "data": d}, default=str) + "\n" for e, d in events))
    assert client.fault_token not in r.text  # the token is never echoed
    names = [e for e, _ in events]
    assert names[-1] == "done"
    plan = next(d for e, d in events if e == "plan")
    workers = [t["worker"] for t in plan["tasks"]]
    content = {"research", "summary"} & set(workers)
    assert content and "data" in workers, f"planner did not split the question: {workers}"
    hits = [d for e, d in events if e == "fault_injected"]
    assert hits and all(h["target"] in FAULT.split(",") and h["injected"] for h in hits)
    wr = [d for e, d in events if e == "worker_result"]
    failed = [w for w in wr if w["agent"] in content]
    data = [w for w in wr if w["agent"] == "data"]
    assert failed and all(w["status"] == "failed" and w.get("injected") for w in failed)
    assert len(failed) == len([t for t in plan["tasks"] if t["worker"] in content])  # worker_error: no retry
    assert data and data[-1]["status"] in ("done", "partial")
    done = [d for e, d in events if e == "done"][0]
    assert done["status"] == "partial", done["status"]
    answer = "".join(d["delta"] for e, d in events if e == "token")
    assert "Incomplete evidence" in answer and "injected test fault" in answer
    assert any(w.title() in answer.split("Incomplete evidence")[-1] for w in content)
    assert any(d["cite_id"].startswith("D") for e, d in events if e == "citation")
    assert not [d for e, d in events if e == "error" and not d.get("recoverable")]
