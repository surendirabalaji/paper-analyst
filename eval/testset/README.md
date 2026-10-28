# Phase 9 fixed test set (`testset_v1.jsonl`)

**Status: DRAFT, awaiting student review. Not frozen; no scoring has been run on it.**

## Composition (36 items)

| Category | n | Used for | Latency category |
|---|---|---|---|
| `rag_single`: one paper, one section | 20 | RAGAS (4 metrics), provider comparison | simple |
| `rag_compare`: two papers | 6 | RAGAS (4 metrics), provider comparison | simple (the planner may still choose 2 tasks; the actual plan is recorded) |
| `data`: corpus metadata (SQL) | 4 | exact-match system check (answer vs SQL result) | simple |
| `multi`: Data → Summary hand-off | 2 | hand-off check; multi-agent latency | multi |
| `system`: out-of-corpus, missing paper, live web, MCP sharing | 4 | behaviour checks (expected behaviour written below each item) | not timed |

RAGAS set: **26 items** (20 single + 6 comparison). Data, multi and system items are excluded from RAGAS because their evidence is SQL rows or live web results, not corpus chunks, or because the expected behaviour is an abstention, which the four metrics do not score meaningfully.

## Construction method

1. **Source chunks (deterministic):** `eval/testset/sample_chunks.py` (seed 599, no network) picks one chunk from each of 20 distinct papers:
   - fixed per-topic quota proportional to the corpus: rag 4, multi-agent 4, benchmarks 3, memory 3, reasoning-and-acting 2, tool use 2, agentic search 1, MCP/frameworks 1;
   - section types rotate method → results → analysis/limitations;
   - chunks must be ≥ 250 tokens, at least 50 % prose lines (drops table dumps and figure-caption lists), and outside the abstract, introduction, related work, appendices and prompt listings;
   - papers used by the Phase 8b development questions (Self-RAG, ReAct, AgentBench, CRAG) are excluded, so the set is held out from the questions the latency work was tuned on.

   The first sampler run (no prose or appendix filter) produced three table- or caption-only chunks; the two filters were added and the sampler rerun before any question was written. Output: `sampled_chunks.jsonl`.
2. **Comparison pairs:** six pairs of corpus papers on the same problem (MemGPT/A-MEM, multiagent debate/MoA, GraphRAG/RAPTOR, RAGAS/ARES, Toolformer/Gorilla, Search-o1/Search-R1). Their reference chunks were found with a paper-scoped `Retriever.search` (k = 3) on a description of the mechanism, then read; only chunks that state the compared mechanism were kept (2–4 per question).
3. **Questions and references:** drafted by Claude (`claude-opus-5-5`, Claude Code, 2026-10-27) from the chunk text only:
   - questions are phrased as a user would ask, naming the paper or method but never the section;
   - every claim in a reference answer restates its reference chunks, with no outside knowledge;
   - `reference_chunk_ids` lists the chunks; all 40 unique ids resolve in the Chroma index (checked).

   Source: `draft_v1.py` writes the JSONL.
4. **Data and multi items:** the reference answers were computed with the SQL stored in `reference_sql` against `data/papers.db`.
5. **Student review (pending):** edit, drop or replace items, then freeze.
6. **Freeze:** after review, the file's SHA-256 is recorded here and in `docs/evaluation.md`, and the file is committed. It is not edited after scoring starts; any later change becomes `testset_v2` with its own results.

## Known limits of this set

- References come from 1–4 chunks. A system answer can be correct and cite other valid chunks of the same paper. Context recall then reads lower than the answer quality, because it measures whether the *retrieved* contexts support the *reference* claims.
- 20 + 6 RAGAS items give coarse means (one sample per question per configuration); report them with the per-item spread, not as precise figures.
- One person drafted the items and the same team reviews them. There is no second annotator.
