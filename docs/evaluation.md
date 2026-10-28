# Evaluation

> **Status: OUTLINE ONLY. No results have been measured yet.** Every number in this file will come from a recorded run in `eval/results/`. Nothing here is a target or an estimate unless labeled as one.

## 1. Fixed test set

- **File:** `eval/testset/testset_v1.jsonl`, frozen 2026-10-28 before any scoring, SHA-256 `50e9c0f3e629e9bc39f6412e72133ab8c925b61f4eeaf22e791680cf1f8f5382`. Construction method and composition: `eval/testset/README.md`.
- **Planned composition:** about 30 questions.
  - 20 single-paper factual questions, stratified across papers and section types (method, results, limitations).
  - 6 cross-paper comparison questions.
  - 4 metadata/code questions (system tests, not RAGAS: their evidence is not chunk text).
- **Construction:** sampled chunks → LLM-drafted question, reference answer and reference chunk IDs → **reviewed and edited by the student**.
- **Separate system tests:** about 5 out-of-corpus questions (abstention / live escalation), failure-injection cases, MCP-sharing cases.

## 2. RAGAS

| Metric | RAGAS class (v0.4.3, to verify) | Mean | n |
|---|---|---|---|
| Faithfulness | | pending | |
| Answer relevance | | pending | |
| Context precision | | pending | |
| Context recall | | pending | |

Configuration to record: RAGAS version, judge model and provider, embedding model, live search disabled for RAGAS runs, aggregation method, command to reproduce.

## 3. Latency

Measured against the deployed Cloud Run URL with `scripts/bench.py`.

| Condition | n | Retrieval ms | LLM ms | TTFT ms | Total ms |
|---|---|---|---|---|---|
| Warm (steady state, p50 / max) | | pending | | | |
| Cold start (first request after idle) | | pending | | | |

## 4. Cost per query

Estimated from recorded token counts × dated provider prices (labeled as an estimate). Embedding, Tavily and RAGAS judge costs are reported separately.

## 5. Provider comparison

Same questions and same retrieved contexts; only the generation provider changes (evaluation-only override).

| Provider / model | Faithfulness | Relevance | TTFT | Gen latency | Est. cost / query |
|---|---|---|---|---|---|
| OpenAI `gpt-6.1-sol` | pending | | | | |
| Gemini `google/gemini-3.8-flash` via OpenRouter | pending | | | | |

Routed-mode provider share across all evaluation queries: pending.

## 6. Limitations

To be written from the actual runs.
