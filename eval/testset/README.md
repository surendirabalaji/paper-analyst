# Phase 9 fixed test set (`testset_v1.jsonl`)

**Status: DRAFT, awaiting student review. Not frozen; no scoring has been run on it.**

## Composition (44 items)

| Category | n | Used for | Latency category |
|---|---|---|---|
| `rag_single`: one paper, one section | 20 | RAGAS (4 metrics), provider comparison | simple |
| `rag_compare`: two papers | 6 | RAGAS (4 metrics), provider comparison | simple (the planner may still choose 2 tasks; the actual plan is recorded) |
| `rag_hard`: 4 paraphrases (no paper name or its own terms), 1 similar-name hard negative, 1 mixed-evidence | 6 | RAGAS (4 metrics), reported also as their own subgroup | simple |
| `data`: corpus metadata (SQL) | 4 | exact-match system check (answer vs SQL result) | simple |
| `multi`: Data → Summary hand-off | 2 | hand-off check; multi-agent latency | multi |
| `system`: out-of-corpus, missing paper, live web, MCP sharing, insufficient evidence, adversarial | 6 | behaviour checks (expected behaviour written below each item) | not timed |

RAGAS set: **32 items** (20 single + 6 comparison + 6 hard). Data, multi and system items are excluded from RAGAS because their evidence is SQL rows or live web results, not corpus chunks, or because the expected behaviour is an abstention, which the four metrics do not score meaningfully.

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
5. **Review round 1 (2026-10-28):** the student supplied an external review of the 36-item draft. Adopted:
   - `rag_hard` items p01–p04, h01, h02;
   - system checks x05 (insufficient evidence) and x06 (adversarial);
   - `required_facts` on every RAGAS and multi item: the minimum a correct answer states, with the reference keeping the optional detail;
   - d04 uses `COUNT(DISTINCT arxiv_id)`. The value is unchanged at 58, since `repos.arxiv_id` is the primary key;
   - x04 relabelled as an architecture-compliance check;
   - corpus snapshot ids recorded below.

   Not adopted:
   - The suggested "Search-R1 vs R1-Searcher" is kept (h01), but as a similar-name test: R1-Searcher has no paper of its own in the corpus and appears only as a TeaRAG baseline.
   - Expansion to ~60 items: about 3 configurations × 32 RAGAS runs already uses most of the $5 budget.
   - A separate oracle-context mode: the four required metrics are run end to end only (below).
6. **Student review (pending):** edit, drop or replace items, then freeze.
7. **Freeze:** after review, the file's SHA-256 is recorded here and in `docs/evaluation.md`, and the file is committed. It is not edited after scoring starts; any later change becomes `testset_v2` with its own results.

## How the set is used (no single aggregate score)

- **End to end only:** the system gets the question alone. `reference_chunk_ids` are never passed to the pipeline; they are ground truth for a non-LLM retrieval diagnostic (was a reference chunk retrieved, and at what rank).
- RAGAS contexts are what the system actually retrieved in that run (every `corpus_search` and section read), not the reference chunks.
- Results are reported per category:
  - RAGAS means for single, comparison and hard items;
  - exact match for Data;
  - required-fact rubric for multi;
  - pass/fail trace assertions for system checks.

## Corpus snapshot the references were computed on

| Item | Value |
|---|---|
| `data/corpus_manifest.json` `generated_at` | 2026-10-27T05:00:32Z (re-ingest: 0 added, 4,521 unchanged) |
| `data/corpus_manifest.json` SHA-256 | `5cf6c026e77d5ff73520c4af50bc589d47dd22ae29d0df51441c19327813b60b` |
| `data/papers.db` SHA-256 | `8e9e5d54504c199d9fe85406e601c63b7d6baf20e865123a366a25bb9a745093` |
| Chroma collection `arxiv_agents` | 4,521 chunks, `text-embedding-3-small`, 1536-d |

If any of these change, re-check the Data references and chunk ids before comparing results.

## Known limits of this set

- References come from 1–4 chunks. A system answer can be correct and cite other valid chunks of the same paper. Context recall then reads lower than the answer quality, because it measures whether the *retrieved* contexts support the *reference* claims.
- 32 RAGAS items give coarse means (one sample per question per configuration); report them with the per-item spread, not as precise figures.
- Claude drafted the items; the student and one external review checked them. There is no independent second annotator.
- The paraphrases p01–p04 reuse the references of s01, s03, s10 and s15, so those pairs also measure robustness to wording on the same evidence.
