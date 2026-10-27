"""Tester Phase 2: real HTTP (uvicorn on a socket) with scripted (MOCKED) LLM + retriever, plus
source-endpoint checks against the REAL read-only Chroma index (no embedding calls). IDs: T2-H-*.

What this proves that in-process tests cannot: the SSE body is flushed through FastAPI/uvicorn as
events are produced (client arrival gaps track the upstream token gaps), framing is valid on the wire,
and a client disconnect cancels the in-flight graph.
"""

import asyncio
import json
import pathlib
import socket
import sys
import threading
import time

import httpx
import pytest
import uvicorn
from langchain_core.messages import AIMessage

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from phase2_fakes import FakeRetriever, FakeRouter, Script, make_settings, plan, tool_call  # noqa: E402

import app.main as main  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT: dict = {}


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(scope="module")
def server():
    """Run the real app on a socket; lifespan builds fakes instead of the real retriever/router."""
    mp = pytest.MonkeyPatch()
    mp.setattr(main, "load_settings", lambda: make_settings())
    mp.setattr(main, "Retriever", lambda s: FakeRetriever())
    mp.setattr(main, "Router", lambda s: FakeRouter(s, _ScriptProxy()))
    port = _free_port()
    cfg = uvicorn.Config(main.app, host="127.0.0.1", port=port, log_level="warning")
    srv = uvicorn.Server(cfg)
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    th.join(5)
    mp.undo()


class _ScriptProxy:
    """Lets each test swap the active script while the server keeps one router instance."""

    def __getattr__(self, k):
        return getattr(SCRIPT["s"], k)

    def __setattr__(self, k, v):
        setattr(SCRIPT["s"], k, v)


def _research_script(**kw):
    return Script(
        planner=plan(("research", "explain alpha", [])),
        worker=[AIMessage(content="", tool_calls=[tool_call("corpus_search", {"query": "alpha retrieval"})]),
                AIMessage(content="Alpha [C1].")],
        **kw,
    )


def parse_frames(raw: str):
    """Strict SSE frame check: every frame is event+single data line or a comment."""
    assert raw.endswith("\n\n"), "stream must end on a frame boundary"
    out = []
    for frame in raw.split("\n\n")[:-1]:
        lines = frame.split("\n")
        if all(l.startswith(":") for l in lines):
            continue
        assert len(lines) == 2 and lines[0].startswith("event: ") and lines[1].startswith("data: "), frame[:200]
        out.append((lines[0][7:], json.loads(lines[1][6:])))
    return out


def test_T2_H1_stream_is_incremental_on_the_wire(server):
    SCRIPT["s"] = _research_script(synth_chunks=[f"w{i} " for i in range(8)] + ["[C1]"], synth_delay=0.4)
    arrivals = []
    raw = ""
    with httpx.Client(timeout=30) as c:
        t0 = time.perf_counter()
        with c.stream("POST", f"{server}/api/chat", json={"query": "How does alpha work?"}) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            assert r.headers.get("x-request-id")
            for text in r.iter_text():
                raw += text
                arrivals.append((time.perf_counter() - t0, raw.count("event: token")))
    events = parse_frames(raw)
    assert events[-1][0] == "done" and events[-1][1]["status"] == "done"
    # time at which the client had seen the first token vs the last token
    first = next(t for t, n in arrivals if n >= 1)
    last = next(t for t, n in arrivals if n >= 9)
    assert last - first > 2.5, f"tokens arrived in a burst ({last - first:.2f}s) -> buffered"
    # at least 5 distinct network reads carried new tokens before done
    counts = sorted({n for _, n in arrivals})
    assert len(counts) >= 6


def test_T2_H2_client_disconnect_cancels_graph(server):
    s = _research_script(synth_chunks=[f"w{i} " for i in range(20)], synth_delay=0.3)
    SCRIPT["s"] = s
    with httpx.Client(timeout=30) as c:
        with c.stream("POST", f"{server}/api/chat", json={"query": "How does alpha work?"}) as r:
            for text in r.iter_text():
                if "event: token" in text:
                    break  # leave after the first token: closes the connection
    time.sleep(3.0)
    assert s.synth_cancelled and not s.synth_completed, "graph kept generating after the client left"


@pytest.mark.parametrize("body,desc", [
    ({"query": ""}, "empty"),
    ({"query": "   \n\t "}, "whitespace"),
    ({"query": "x" * 2001}, "too long"),
    ({"query": "ok", "k": 2}, "k below min"),
    ({"query": "ok", "k": 11}, "k above max"),
    ({"query": "ok", "k": "lots"}, "k not int"),
    ({}, "missing query"),
    ({"query": None}, "null query"),
])
def test_T2_H3_invalid_requests_fail_fast_without_stack_traces(server, body, desc):
    r = httpx.post(f"{server}/api/chat", json=body, timeout=10)
    assert r.status_code == 422, desc
    assert "Traceback" not in r.text and "File \"" not in r.text


def test_T2_H4_non_json_body_is_422(server):
    r = httpx.post(f"{server}/api/chat", content=b"not json", headers={"content-type": "application/json"}, timeout=10)
    assert r.status_code == 422 and "Traceback" not in r.text


def test_T2_H5_unicode_round_trip_on_the_wire(server):
    SCRIPT["s"] = _research_script(synth_chunks=["naïve ", "检索 ", "τ-bench ", "😀 ", "[C1]"])
    with httpx.Client(timeout=30) as c:
        r = c.post(f"{server}/api/chat", json={"query": "Résumé — τ-bench 检索?"})
    events = parse_frames(r.content.decode("utf-8"))
    text = "".join(p["delta"] for e, p in events if e == "token")
    assert text == "naïve 检索 τ-bench 😀 [C1]"


def test_T2_H6_static_page_and_assets(server):
    r = httpx.get(f"{server}/", timeout=10)
    assert r.status_code == 200 and "<script src=\"/static/app.js\">" in r.text
    for p in ("/static/app.js", "/static/styles.css"):
        assert httpx.get(server + p, timeout=10).status_code == 200
    for p in ("/static/../main.py", "/static/%2e%2e/main.py", "/static/..%2fconfig.py"):
        rr = httpx.get(server + p, timeout=10)
        assert rr.status_code == 404 and "load_settings" not in rr.text


# ---------------------------------------------------------------- real index, source endpoint

@pytest.fixture(scope="module")
def real_client():
    from fastapi.testclient import TestClient

    from app.config import load_settings
    from app.rag import Retriever

    if not (ROOT / "data" / "chroma_index").exists():
        pytest.skip("real index not built")
    s = load_settings()
    c = TestClient(main.app)  # no lifespan: set state directly (Retriever does not embed on construction)
    main.app.state.settings = s
    main.app.state.retriever = Retriever(s)
    return c


def test_T2_H7_source_endpoint_resolves_real_chunks_exactly(real_client):
    import sqlite3
    from ingest.store import open_client

    col = open_client(ROOT / "data" / "chroma_index").get_collection("arxiv_agents")
    got = col.get(where={"doc_id": "2305.18323v1"}, include=["documents", "metadatas"], limit=5)
    for cid, doc, md in zip(got["ids"], got["documents"], got["metadatas"]):
        enc = cid.replace(":", "%3A")
        r = real_client.get(f"/api/source/chunk/{enc}")
        assert r.status_code == 200, cid
        j = r.json()
        assert j["chunk_id"] == cid and j["text"] == doc and j["doc_id"] == "2305.18323v1"
        assert j["section_path"] == md["section_path"] and j["html_anchor_url"] == md["html_anchor_url"]
    db = sqlite3.connect(f"file:{ROOT / 'data' / 'papers.db'}?mode=ro", uri=True)
    title = db.execute("select title from papers where doc_id=?", ("2305.18323v1",)).fetchone()[0]
    assert real_client.get(f"/api/source/chunk/{got['ids'][0]}").json()["title"] == title


@pytest.mark.parametrize("bad", [
    "2305.18323v1::S1::999",  # well-formed but absent
    "2305.18323v9::S1::000",  # wrong version
    "..%2F..%2Fetc%2Fpasswd",
    "%2e%2e%2f%2e%2e%2fapp%2fconfig.py",
    "2305.18323v1::S1::000%00",
    "2305.18323v1::S1::000' OR 1=1",
    "x" * 500,
])
def test_T2_H8_source_endpoint_rejects_unknown_and_traversal(real_client, bad):
    r = real_client.get(f"/api/source/chunk/{bad}")
    assert r.status_code == 404
    assert "root:" not in r.text and "load_settings" not in r.text
