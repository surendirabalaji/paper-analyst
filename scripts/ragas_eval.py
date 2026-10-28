#!/usr/bin/env python3
"""Phase 9 RAGAS scoring of recorded runs (from scripts/run_eval.py) against the frozen test set.

Metrics (ragas 0.4.3, `ragas.metrics.collections`), all scored per item, end to end:
  faithfulness       Faithfulness(user_input, response, retrieved_contexts)
  answer_relevancy   AnswerRelevancy(user_input, response)    - the PDF's "answer relevance"; strictness 3,
                     embeddings text-embedding-3-small
  context_precision  ContextPrecision(user_input, reference, retrieved_contexts)  - reference-based variant
  context_recall     ContextRecall(user_input, retrieved_contexts, reference)

Answer = the streamed `token` text. Contexts = every chunk the workers actually received in that run (each
`retrieval` event: corpus_search hits, paper outlines and section reads; web results by their snippet),
de-duplicated, in first-seen order, which is the order context precision is rank-sensitive to. Reference
answers and chunk ids come from the test set; the chunk ids are never sent to the system.
`--max-contexts N` keeps only the first N contexts (recorded in the output when used).

Judge: OPENAI_MODEL (gpt-6.1-sol) with ragas' reasoning-model parameter mapping applied explicitly (same fix
as scripts/faithfulness_check.py). Judge token usage and an estimated cost (dated price table in
app/llm/policy.py) are recorded from the HTTP responses. Writes OUT/ragas.json (rows + summary).

Usage: .venv/bin/python scripts/ragas_eval.py RUN_DIR [--ids s01 ...] [--metrics faithfulness ...]
                                              [--max-contexts N] [--concurrency 4]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openai import AsyncOpenAI  # noqa: E402
from ragas.embeddings import OpenAIEmbeddings  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness  # noqa: E402

from app.config import load_settings  # noqa: E402
from app.llm.policy import estimate_cost  # noqa: E402
from ingest.store import open_client  # noqa: E402

EVAL_MAX_COMPLETION_TOKENS = 16000
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
EMBED_PRICE_PER_1M = 0.02  # text-embedding-3-small, data/corpus_manifest.json (checked 2026-10-26)


class Usage:
    def __init__(self):
        self.chat_in = self.chat_out = self.chat_cached = self.reasoning = self.calls = self.embed_tokens = 0

    async def hook(self, response: httpx.Response) -> None:
        await response.aread()
        try:
            u = response.json().get("usage") or {}
        except Exception:
            return
        if "completion_tokens" in u:
            self.calls += 1
            self.chat_in += u.get("prompt_tokens", 0)
            self.chat_out += u.get("completion_tokens", 0)
            self.chat_cached += (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
            self.reasoning += (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
        elif "prompt_tokens" in u:
            self.embed_tokens += u.get("prompt_tokens", 0)

    def snapshot(self) -> dict:
        return dict(vars(self))


def contexts_of(events: list[dict], collection) -> list[str]:
    """Every piece of evidence the workers received, in arrival order: `retrieval` events (paper chunks,
    SQL rows) and `web_search` events (Tavily results, which never appear in `retrieval`; tester P9-D1)."""
    out, seen, chunk_ids = [], set(), []
    order = []
    for e in events:
        if e["event"] == "retrieval":
            items = e["data"].get("chunks", [])
        elif e["event"] == "web_search" and e["data"].get("status") == "ok":
            items = [{**r, "type": "web"} for r in e["data"].get("results", [])]
        else:
            continue
        for c in items:
            key = c.get("chunk_id") or c.get("url") or c.get("source_url") or c.get("cite_id")
            if not key or key in seen:
                continue
            seen.add(key)
            order.append(c)
            if c.get("type") == "paper_chunk" and c.get("chunk_id"):
                chunk_ids.append(c["chunk_id"])
    texts = {}
    if chunk_ids:
        got = collection.get(ids=chunk_ids, include=["documents"])
        texts = dict(zip(got["ids"], got["documents"]))
    for c in order:
        if c.get("type") == "paper_chunk":
            if c.get("chunk_id") in texts:
                out.append(texts[c["chunk_id"]])
        elif c.get("type") == "web":
            out.append(f"{c.get('title', '')}: {c.get('snippet', '')}")
        elif c.get("rows") is not None:
            out.append(f"SQL: {c.get('sql', '')}\ncolumns: {c.get('columns')}\nrows: {c.get('rows')}")
    return out


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--metrics", nargs="*", default=METRICS)
    ap.add_argument("--max-contexts", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default="ragas.json")
    ap.add_argument("--merge", action="store_true",
                    help="rescore only --ids and merge them into the existing OUT file (summary recomputed over all rows)")
    a = ap.parse_args()

    s = load_settings()
    run_dir = Path(a.run_dir)
    meta = json.loads((run_dir / "results.json").read_text())
    tests = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "eval/testset/testset_v1.jsonl").read_text().splitlines() if l.strip()}
    collection = open_client(s.chroma_dir).get_collection(s.chroma_collection)

    usage = Usage()
    http = httpx.AsyncClient(timeout=httpx.Timeout(180, connect=20), event_hooks={"response": [usage.hook]})
    client = AsyncOpenAI(api_key=s.openai_api_key, http_client=http, max_retries=3)
    llm = llm_factory(s.openai_model, client=client)
    # ragas 0.4.3 detects reasoning models by an integer version ("gpt-6"), so "gpt-6.1-sol" still gets
    # max_tokens, which the API rejects. Apply ragas' own reasoning-model mapping explicitly.
    args = getattr(llm, "model_args", {})
    args.pop("max_tokens", None)
    args["max_completion_tokens"] = EVAL_MAX_COMPLETION_TOKENS
    args["temperature"] = 1.0
    args.pop("top_p", None)
    emb = OpenAIEmbeddings(client=client, model=s.embedding_model)
    scorers = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=emb),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }

    rows_in = [r for r in meta["results"] if tests.get(r["id"], {}).get("ragas")]
    if a.ids:
        rows_in = [r for r in rows_in if r["id"] in a.ids]
    sem = asyncio.Semaphore(a.concurrency)

    async def score(r: dict) -> dict:
        t = tests[r["id"]]
        path = run_dir / r["raw"]
        events = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        answer = "".join(e["data"]["delta"] for e in events if e["event"] == "token")
        ctx = contexts_of(events, collection)
        n_all = len(ctx)
        if a.max_contexts:
            ctx = ctx[: a.max_contexts]
        row = {"id": r["id"], "run": r["run"], "category": t["category"], "status": r.get("status"),
               "n_contexts": len(ctx), "n_contexts_all": n_all, "answer_chars": len(answer)}
        if not answer.strip():
            row["error"] = "empty answer"
            return row
        ctxs = ctx or ["(no retrieved context)"]
        async with sem:
            for m in a.metrics:
                t0 = time.perf_counter()
                kw = {"user_input": t["question"]}
                if m in ("faithfulness", "answer_relevancy"):
                    kw["response"] = answer
                if m != "answer_relevancy":
                    kw["retrieved_contexts"] = ctxs
                if m in ("context_precision", "context_recall"):
                    kw["reference"] = t["reference"]
                try:
                    row[m] = round(float((await scorers[m].ascore(**kw)).value), 4)
                except Exception as e:  # recorded, never invented
                    row[m] = None
                    row.setdefault("errors", {})[m] = f"{type(e).__name__}: {str(e)[:200]}"
                row[f"{m}_s"] = round(time.perf_counter() - t0, 1)
        print(json.dumps({k: row[k] for k in row if k in ("id", "n_contexts", *a.metrics, "errors")}), flush=True)
        return row

    t_start = time.perf_counter()
    rows = await asyncio.gather(*(score(r) for r in rows_in))
    await http.aclose()
    prev_cost, rescored = 0.0, []
    if a.merge and (run_dir / a.out).exists():
        prev = json.loads((run_dir / a.out).read_text())
        new_ids = {(r["id"], r["run"]) for r in rows}
        rescored = sorted(f"{i}.run{n}" for i, n in new_ids)
        old_by = {(r["id"], r["run"]): r for r in prev["rows"]}
        # keep the previous values of metrics not rescored in this call; rescored metrics replace them
        rows = [{**old_by.get((r["id"], r["run"]), {}), **r, "rescored_metrics": list(a.metrics)} for r in rows]
        rows = [r for r in prev["rows"] if (r["id"], r["run"]) not in new_ids] + rows
        rows.sort(key=lambda r: [x["id"] for x in meta["results"]].index(r["id"]) if any(x["id"] == r["id"] for x in meta["results"]) else 0)
        prev_cost = prev["summary"].get("judge_cost_usd_est", 0.0)

    summary_metrics = [m for m in METRICS if any(m in r for r in rows)]

    def summ(sel: list[dict]) -> dict:
        out = {"n_items": len(sel)}
        for m in summary_metrics:
            vals = [x[m] for x in sel if x.get(m) is not None]
            out[m] = {"n": len(vals), "n_failed": len(sel) - len(vals),
                      "mean": round(statistics.mean(vals), 4) if vals else None,
                      "median": round(statistics.median(vals), 4) if vals else None,
                      "min": min(vals) if vals else None, "max": max(vals) if vals else None}
        return out

    u = usage.snapshot()
    judge_cost = estimate_cost(s.openai_model, u["chat_in"], u["chat_out"], u["chat_cached"])
    summary = {
        "run_dir": str(run_dir), "label": meta.get("label"), "testset_sha256": meta.get("testset_sha256"),
        "ragas_version": "0.4.3", "judge_model": s.openai_model, "judge_provider": "openai",
        "embeddings": "text-embedding-3-small", "answer_relevancy_strictness": 3,
        "context_precision_variant": "ContextPrecision (reference-based)", "max_contexts": a.max_contexts,
        "aggregation": "arithmetic mean over items with a score; failed items counted separately, never imputed",
        "all": summ(rows),
        "by_category": {c: summ([x for x in rows if x["category"] == c]) for c in sorted({x["category"] for x in rows})},
        "judge_usage": u,
        "judge_cost_usd_est": round(prev_cost + (judge_cost or 0) + u["embed_tokens"] * EMBED_PRICE_PER_1M / 1e6, 4),
        "contexts": "retrieval + web_search events (P9-D1 fix)",
        **({"rescored_after_P9_D1": rescored} if rescored else {}),
        "wall_s": round(time.perf_counter() - t_start, 1),
    }
    (run_dir / a.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
