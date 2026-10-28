"""Phase 9 (tester P9-D1): RAGAS contexts must include web results, which arrive in `web_search` events,
not in `retrieval` events. Offline: fake events and a fake Chroma collection."""
import importlib.util
import json
import pathlib
import sys

import pytest

pytest.importorskip("ragas")  # requirements-eval.txt
ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("ragas_eval", ROOT / "scripts/ragas_eval.py")
ragas_eval = importlib.util.module_from_spec(spec)
sys.modules["ragas_eval"] = ragas_eval
spec.loader.exec_module(ragas_eval)


class FakeCollection:
    def get(self, ids, include):
        return {"ids": ids, "documents": [f"text of {i}" for i in ids]}


def ev(name, **data):
    return {"event": name, "data": data}


def test_web_search_results_become_contexts_in_arrival_order_without_duplicates():
    events = [
        ev("retrieval", chunks=[{"type": "paper_chunk", "chunk_id": "a::S1::000", "cite_id": "C1"}]),
        ev("web_search", status="ok", results=[{"cite_id": "W1", "url": "https://x/1", "title": "T1", "snippet": "S1"},
                                               {"cite_id": "W2", "url": "https://x/2", "title": "T2", "snippet": "S2"}]),
        ev("web_search", status="error", results=[{"cite_id": "W9", "url": "https://x/9", "title": "bad", "snippet": "no"}]),
        ev("web_search", status="ok", results=[{"cite_id": "W1", "url": "https://x/1", "title": "T1", "snippet": "S1"}]),
        ev("retrieval", chunks=[{"type": "paper_chunk", "chunk_id": "a::S1::000", "cite_id": "C1"},
                                {"type": "paper_chunk", "chunk_id": "b::S2::001", "cite_id": "C2"}]),
    ]
    assert ragas_eval.contexts_of(events, FakeCollection()) == [
        "text of a::S1::000", "T1: S1", "T2: S2", "text of b::S2::001"]


@pytest.mark.parametrize("run,item", [("routed-deployed", "c06"), ("forced-gemini-local", "c06"), ("forced-gemini-local", "p02")])
def test_recorded_runs_with_web_citations_get_web_contexts(run, item):
    path = ROOT / f"eval/results/phase9/{run}/{item}.run1.events.jsonl"
    if not path.exists():
        pytest.skip("Phase 9 evidence not present")
    events = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    cited_web = {e["data"]["snippet"] for e in events if e["event"] == "citation" and e["data"].get("type") == "web"}
    ctx = ragas_eval.contexts_of(events, FakeCollection())
    assert cited_web and all(any(s in c for c in ctx) for s in cited_web)
