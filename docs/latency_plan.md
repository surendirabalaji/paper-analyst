# Latency plan (Phase 8b, deferred: start after Phase 8, before Phase 9)

Discussion recorded 2026-10-27 after Phase 6. The student decided to do this work **after Phase 8** (UI polish) and before Phase 9 (evaluation), so Phase 9 measures the improved system.

## Student goals and constraints

- Targets set by the student: **cold-start TTFT under 20–25 s**, **warm TTFT around 1.5–2 s**. These come from the PDF §5 reference deployment, which the PDF gives "for reference" only. The PDF sets no latency threshold, and we must not report the reference figures as our own.
- **No rule-based query classifier.** The LLM planner stays and decides the plan.
- **No semantic cache.** On an unseen question it gives no speedup and a misleading latency figure; near-matches can return wrong answers; and it conflicts with the stateless design.
- Speculative decoding was considered and is not applicable. It runs inside the inference server (we call hosted APIs), and our slow calls emit only 21–40 tokens each, so the time is per-call overhead, not generation speed.

## Baseline (measured 2026-10-27, rev `paper-analyst-00010-vpv`, one sample each)

Question: "How does Self-RAG decide when to retrieve?" Raw events are in `eval/results/ttft-cold-warm-20261027T2255Z/`.

| | Retrieval | LLM (sum) | TTFT (client) | Total (client) |
|---|---|---|---|---|
| Cold (instance scaled to zero since 22:35 UTC) | 34 ms | 27.1 s | 52.0 s (first SSE byte at 27.0 s) | 54.2 s |
| Warm (same question, right after) | 34 ms | 25.2 s | 22.4 s | 25.4 s |

Warm timeline:

| Step | Finished at | Duration |
|---|---|---|
| Planner (Gemini, 366 output tokens) | 5.3 s | 5.1 s |
| Summary round 1 → `get_paper_outline` | 7.1 s | 1.9 s |
| Summary round 2 → `get_paper_section` | 9.1 s | 2.0 s |
| Summary round 3 → `get_paper_section` | 11.5 s | 2.4 s |
| Summary round 4 → `get_paper_section` | 13.7 s | 2.2 s |
| Summary final write-up (OpenAI, capability: > 3k section tokens; 319 output tokens) | 20.7 s | 7.0 s |
| Synthesis first token (Gemini) | 22.4 s | 1.7 s |
| Synthesis complete | 25.4 s | |

Tool execution took 9–24 ms per call. Prompt size grew from 1,076 → 2,364 → 3,934 → 4,979 → 5,752 input tokens across the worker calls.

## Why it is slow

1. Six LLM calls run in sequence before the first answer token.
2. The worker picks one tool per LLM round. Each round costs about 2 s of fixed per-call overhead for 21–40 output tokens.
3. The prompt grows every round, because tool results stay in context.
4. Double writing: the worker writes findings (7.0 s on OpenAI), then synthesis rewrites them.
5. The planner's output is verbose: 366 tokens took 5.2 s, against 2.0–2.5 s when its output is short.
6. Possibly the OpenRouter → Google AI Studio hop (unmeasured).
7. Cold start adds about 27 s. The image is 1.86 GB (425 MB compressed) and includes ragas, datasets and scipy, which the app never imports; imports take about 13 s; the Chroma index takes 5–7 s to open on Cloud Run against 0.4 s locally (cause unknown).

Retrieval and the app's own code are not the cause.

**Hard limit:** with an LLM planner that decides before any answer text is shown, warm TTFT is at least about planner + synthesis first token, roughly **3–4 s**. Only speculation (option 9) overlaps the two and gets near 2 s.

## Options (estimates; verify by measurement)

| # | Option | Saves (warm) | Risk |
|---|---|---|---|
| 1 | Leaner planner: shorter instructions and summary, minimal effort, tighter schema | ~2–3 s | Low (recheck plan quality with the planner probe) |
| 2 | Planner also names the target papers and the first search queries, so the worker's first round needs no LLM | ~2 s | Low to medium |
| 3 | Batch tool calls (all needed sections in one round) | ~4 s | Low |
| 4 | Drop the worker write-up on single-task plans; synthesis works from the gathered evidence | ~7 s | Medium (check RAGAS faithfulness) |
| 5 | Single-task plans: the worker streams the final answer under its own prompt | ~1.5–7 s (overlaps 4) | Medium (document that the synthesizer runs only on multi-task plans) |
| 6 | Speculative retrieval in parallel with the planner (traced; discarded on Data plans) | ~0.3–0.5 s | Low |
| 7 | `minimal` reasoning effort for Gemini (Phase 4: TTFT 1.9 s vs 2.5 s at low) | ~0.5 s per call | Low to medium |
| 8 | Parallel dispatch of independent tasks (LangGraph `Send`) | 0 on simple questions, ~5–10 s on two-task questions | Low |
| 9 | Speculative answer path: run retrieval and a buffered synthesis while the planner runs; release the buffer only if the plan is a single Research/Summary task, otherwise cancel; never show text the planner did not approve | TTFT ≈ planner time, ~2–3 s | Medium to high |
| 10 | Cold start: runtime-only requirements (move eval deps out, P3-N4), lazy imports, profile the index open | ~5–12 s of container start (unverified) | Low |
| 11 | `min-instances=1` (still measure and report a real cold start separately) | No cold starts for users | Ongoing cost (verify the rate) |

## Packages (simple single-paper question)

| Package | Options | Warm TTFT (expected) | Cold TTFT (expected) |
|---|---|---|---|
| A. Conservative | 1, 3, 7, 10 | ~10–13 s | ~25–35 s |
| B. Restructured | A + 2, 4, 6 | ~4–6 s | ~18–28 s |
| **C. Single-agent streaming (recommended)** | B + 5 | ~3.5–5 s | ~17–26 s |
| D. Speculative | C + 9 | ~2–3 s | ~15–24 s |

Multi-agent questions (e.g. Data → Summary) use options 1, 2, 3, 7 and 8: about 8–14 s warm, from about 28–35 s today.

**Recommendation recorded:** build package C; add D only if measurements show C missing the target. Report simple and multi-agent questions as separate latency categories, and explain the planner + synthesis limit.

**Student decision (2026-10-28):** package B (options 1, 2, 3, 4, 6, 7, 10). Option 8 is held back until the student approves it.

## Proposed exit check for Phase 8b

- Warm TTFT on simple questions ≤ ~4–5 s (package C), as the median of ≥ 5 runs over several questions.
- Cold TTFT ≤ 25 s after a real scale-to-zero.
- Multi-agent questions measured and reported as their own category.
- No regression in citation resolution, plan quality (planner probe), the Data → Summary hand-off, MCP tool sharing, or the use of both providers in ordinary queries.
- The tester runs in phase mode.

## Watch items

- RAGAS faithfulness may drop if fewer sections are read or the worker's self-check is removed. Measure before and after on the same questions.
- Fewer LLM calls means fewer routing decisions. Confirm that both providers still appear under ordinary use (rubric 15%).
- Orchestration evidence (rubric 25%): the planner, supervisor transitions, hand-offs and agent identities must stay visible in the trace.

## Results (package B, measured 2026-10-28)

Method: `scripts/measure_latency.py` (client-side wall clock; TTFT = first `token` SSE event), fixed question set (5 simple, 2 "multi"), warm runs after one unrecorded warm-up request. Faithfulness: `scripts/faithfulness_check.py` (RAGAS 0.4.3 `Faithfulness`, evaluator `gpt-6.1-sol`, contexts = the sources each answer cites). Raw events and summaries are in `eval/results/phase8b/`.

| | Before (rev 00015, warm) | After (rev 00016, warm) |
|---|---|---|
| Simple questions, TTFT median | 18.1 s (n = 5, 15.9–23.3 s) | **6.35 s** (n = 10, 4.6–8.3 s) |
| Simple questions, total median | 23.5 s | 12.3 s |
| Data → Summary ("2025 cs.MA … two most recent"), TTFT | 35.3 s (n = 1) | 18.2 / 19.2 s (n = 2) |
| Self-RAG vs CRAG comparison, TTFT | 20.9 s (n = 1) | 4.9 / 5.8 s (n = 2; planned as one Research task) |
| Cost per simple question | $0.039–0.050 | $0.014–0.032 |
| RAGAS faithfulness, mean | 0.959 (n = 7) | 0.985 (n = 14) |

- The warm target for package B (~4–6 s) is met at the median for single-paper Summary questions (4.6–7.6 s) and close for Research questions (5.1–8.3 s).
- Where the time goes now (simple question): planner ~2–3 s, one worker round ~2.5–4.5 s, first synthesis token ~1–3 s. The planner at `low` effort occasionally reasons (200–240 reasoning tokens, 10–12 s; seen locally in 3 of 21 calls, not in the deployed runs).
- Multi-agent plans keep worker findings and sequential dispatch, so they remain ~18–19 s; option 8 (parallel dispatch) is what would reduce them.
- Both providers still appear under ordinary use: Research questions and multi-task plans synthesize on OpenAI (≥ 3 papers or 2 worker results); short single-paper Summary questions stay on Gemini.
- Cold start: deferred to the next phase by the student (2026-10-28); not measured on rev 00016.
- Limits: one region, small samples (n = 10 simple), a single client location; faithfulness uses one evaluator model and cited sources only, so it says nothing about recall.
