#!/usr/bin/env python3
"""RAGAS faithfulness over recorded runs (Phase 8b before/after check; not the Phase 9 evaluation).

For every `*.events.jsonl` in the given run directories: the answer is the streamed `token` text, and the
contexts are the texts of the sources the answer cites (`citation` events): paper chunks read from the local
Chroma index by chunk id, web results by their snippet, SQL results by their rows. RAGAS `Faithfulness`
(ragas.metrics.collections, pinned 0.4.3) then scores how many answer claims those contexts support.
Evaluator: OPENAI_MODEL (OpenAI direct). Writes `faithfulness.json` into each run directory.

Usage: .venv/bin/python scripts/faithfulness_check.py DIR [DIR ...]
Needs requirements-eval.txt installed and OPENAI_API_KEY in the environment or .env.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openai import AsyncOpenAI  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics.collections import Faithfulness  # noqa: E402

from app.config import load_settings  # noqa: E402
from ingest.store import open_client  # noqa: E402

EVAL_MAX_COMPLETION_TOKENS = 16000


def load_run(path: Path, collection) -> dict:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    query = next((e["data"].get("query") for e in events if e["event"] == "meta" and e["data"].get("query")), None)
    answer = "".join(e["data"]["delta"] for e in events if e["event"] == "token")
    contexts = []
    for e in events:
        if e["event"] != "citation":
            continue
        d = e["data"]
        if d.get("type") == "paper_chunk" and d.get("chunk_id"):
            got = collection.get(ids=[d["chunk_id"]], include=["documents"])
            if got["ids"]:
                contexts.append(f"[{d['cite_id']}] {got['documents'][0]}")
        elif d.get("type") == "web":
            contexts.append(f"[{d['cite_id']}] {d.get('title', '')}: {d.get('snippet', '')}")
        elif d.get("rows") is not None:
            contexts.append(f"[{d['cite_id']}] SQL: {d.get('sql', '')}\ncolumns: {d.get('columns')}\nrows: {d.get('rows')}")
    return {"query": query, "answer": answer, "contexts": contexts}


async def main(dirs: list[str]) -> int:
    s = load_settings()
    collection = open_client(s.chroma_dir).get_collection(s.chroma_collection)
    llm = llm_factory(s.openai_model, client=AsyncOpenAI(api_key=s.openai_api_key))
    # ragas 0.4.3 detects reasoning models by an integer version ("gpt-6"), so "gpt-6.1-sol" still gets
    # max_tokens, which the API rejects. Apply ragas' own reasoning-model mapping explicitly.
    args = getattr(llm, "model_args", {})
    args.pop("max_tokens", None)
    args["max_completion_tokens"] = EVAL_MAX_COMPLETION_TOKENS  # room for reasoning plus the statement list
    args["temperature"] = 1.0
    args.pop("top_p", None)
    metric = Faithfulness(llm=llm)
    for d in dirs:
        rows = []
        res = Path(d) / "results.json"  # written by scripts/measure_latency.py: raw file -> question
        queries = {r["raw"]: r["query"] for r in json.loads(res.read_text())["results"]} if res.exists() else {}
        for f in sorted(Path(d).glob("*.run*.events.jsonl")):
            run = load_run(f, collection)
            if not run["answer"].strip():
                continue
            query = run["query"] or queries.get(f.name) or f.name.split(".run")[0]
            try:
                score = (await metric.ascore(user_input=query, response=run["answer"],
                                             retrieved_contexts=run["contexts"] or ["(no cited sources)"])).value
            except Exception as e:  # recorded, never invented
                score, err = None, f"{type(e).__name__}: {str(e)[:160]}"
            else:
                err = None
            rows.append({"file": f.name, "n_contexts": len(run["contexts"]), "faithfulness": score, "error": err})
            print(json.dumps(rows[-1]), flush=True)
        vals = [r["faithfulness"] for r in rows if r["faithfulness"] is not None]
        summary = {"dir": d, "evaluator": s.openai_model, "metric": "ragas.metrics.collections.Faithfulness (0.4.3)",
                   "n": len(vals), "n_failed": len(rows) - len(vals),
                   "mean": round(statistics.mean(vals), 4) if vals else None,
                   "median": round(statistics.median(vals), 4) if vals else None}
        (Path(d) / "faithfulness.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
        print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
