#!/usr/bin/env python3
"""Deterministic source-chunk sampler for the Phase 9 test set (step 1 of the construction method).

Picks one chunk from each of 20 distinct papers: a fixed per-topic quota, papers drawn with a seeded RNG,
section types rotated method -> results -> limitations/analysis so the set is not all method
descriptions, and chunks must be mostly prose (MIN_PROSE) and outside appendices. Papers used as Phase 8b development questions are excluded, so the test set is held out from
the questions the latency work was tuned on. Writes eval/testset/sampled_chunks.jsonl (id, paper, section, text).

Usage: .venv/bin/python eval/testset/sample_chunks.py   (no network, no LLM calls)
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.config import load_settings  # noqa: E402
from ingest.store import open_client  # noqa: E402

SEED = 599
QUOTA = {"rag": 4, "multi-agent-systems": 4, "agent-benchmarks-and-evaluation": 3,
         "agent-memory-and-architecture": 3, "reasoning-and-acting": 2, "tool-use": 2,
         "agentic-search-and-deep-research": 1, "mcp-and-frameworks": 1}
# Phase 8b development questions (latency + faithfulness runs) used these papers.
EXCLUDE = {"2310.11511", "2210.03629", "2308.03688", "2401.15884"}
TYPES = {
    "method": re.compile(r"method|approach|framework|design|architecture|model|algorithm|overview|system|pipeline", re.I),
    "results": re.compile(r"result|experiment|evaluation|performance|main|benchmark|finding", re.I),
    "analysis": re.compile(r"limitation|analysis|discussion|ablation|error|failure|challenge|conclusion", re.I),
}
SKIP = re.compile(r"abstract|introduction|related|reference|acknowledg|appendix|additional|prompt|example|impact|ethic", re.I)
MIN_TOKENS = 250
APPENDIX_ID = re.compile(r"A\d")  # arXiv HTML appendix sections are A1, A2, ...
MIN_PROSE = 0.5  # share of non-empty lines with >= 8 words; drops table dumps and figure-caption lists


def prose_share(text: str) -> float:
    lines = [ln for ln in text.splitlines()[1:] if ln.strip()]  # skip the title/section prefix line
    return sum(len(ln.split()) >= 8 for ln in lines) / max(len(lines), 1)


def main() -> int:
    s = load_settings()
    col = open_client(s.chroma_dir).get_collection(s.chroma_collection)
    papers = json.loads((ROOT / "data/corpus_manifest.json").read_text())["papers"]
    rng = random.Random(SEED)
    got = col.get(include=["metadatas", "documents"])
    by_doc: dict[str, list] = {}
    for cid, m, d in zip(got["ids"], got["metadatas"], got["documents"]):
        by_doc.setdefault(m["doc_id"], []).append((cid, m, d))
    order = list(TYPES)
    out, k = [], 0
    for topic, n in QUOTA.items():
        pool = sorted(p["doc_id"] for p in papers if p["topic"] == topic and p["fulltext"] and p["arxiv_id"] not in EXCLUDE)
        rng.shuffle(pool)
        taken = 0
        for doc in pool:
            if taken == n:
                break
            want = order[k % len(order)]
            cands = [c for c in sorted(by_doc.get(doc, []), key=lambda c: c[0])
                     if c[1]["n_tokens"] >= MIN_TOKENS and not SKIP.search(c[1]["section_path"]) and not APPENDIX_ID.match(c[1]["section_id"])
                     and TYPES[want].search(c[1]["section_path"]) and prose_share(c[2]) >= MIN_PROSE]
            if not cands:
                continue
            cid, m, d = rng.choice(cands)
            out.append({"chunk_id": cid, "arxiv_id": m["arxiv_id"], "title": m["title"], "topic": topic,
                        "section_type": want, "section_path": m["section_path"], "text": d})
            taken += 1
            k += 1
    path = ROOT / "eval/testset/sampled_chunks.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
    print(f"{len(out)} chunks -> {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
