# Build Plan: Phase-wise Development

We build one phase at a time. Each phase ends with an exit check. The `assignment2-tester` subagent verifies the check in phase mode, and a phase is marked done only on PHASE PASS. Work then stops for review before the next phase starts. Design details are in [design_proposal.md](design_proposal.md). Requirement tracking is in [compliance.md](compliance.md).

**Status values:** not started · in progress · done · blocked

| Phase | Goal | What we build | Exit check | Status |
|---|---|---|---|---|
| **0. Setup check** | Prove every external dependency works before writing app code | Python 3.13 venv; install pinned `requirements.txt`; one streaming test call each to OpenAI and Gemini via OpenRouter (confirm exact model IDs, tool calling, usage fields); Tavily MCP `list_tools` with the key as a query parameter; inspect 3 sample arXiv HTML pages | All three services respond; packages install on 3.13; model IDs confirmed and recorded | done |
| **1. Corpus** | Build the data the agents search | arXiv selection (≈80 papers), HTML parser, section-aware chunking, Chroma index, SQLite metadata (`papers`, `authors`, `categories`, `repos` as links only; repo snapshots deferred to Phase 6 by student decision), `corpus_manifest.json` | Paper and chunk counts recorded; re-running ingest adds 0 chunks | done |
| **2. First working version** | Prove the full path end to end with one worker | FastAPI SSE endpoint, minimal HTML page, graph: planner → supervisor → Research → synthesize; token streaming; `C#` citations; `/api/source/chunk` | Browser shows a streamed answer with resolvable citations and a trace | done |
| **3. Early deploy** | Find container and Cloud Run problems early | Multi-stage Dockerfile (python:3.13-slim, non-root, `$PORT`), deploy script, first Cloud Run deploy | Public URL answers a corpus question after a cold start | done |
| **4. Routing** | Real two-provider routing | Hybrid router, `llm_route` and `fallback` events, Routing tab | Two ordinary questions pick different providers, with visible rationale | done |
| **5. Tavily + Summary** | Live retrieval and MCP tool sharing | Tavily Remote MCP client with redaction; Summary agent and its tools; shared `tavily_search` | Trace shows both Research and Summary calling `tavily_search` in different contexts | done |
| **6. Data (+ Code)** | Remaining workers and real hand-offs | Data agent (read-only SQL, `D#` citations); Code agent over repo snapshots (`R#` citations) if time allows | A Data → Summary hand-off works; Research → Code if Code is built (Code not built, decision recorded) | done |
| **7. Failure handling** | Graph survives failures | Worker wrappers, retries, budgets, fault-injection flag, tests | A forced worker failure gives a partial answer, not a crash | done |
| **8. UI polish** | Meet all four frontend requirements clearly | Trace badges, source viewer, error states, Unicode handling, single-turn banner; redeploy | Deployed UI shows streaming, per-turn routing, readable trace and linked citations | done (tester PHASE PASS, rev 00013; P8-D1, P8-D2 Minor open) |
| **8b. Latency** | Cut TTFT (decided 2026-10-27; starts after Phase 8, before Phase 9) | Package from [latency_plan.md](latency_plan.md) (recommended C: leaner planner, planner-named papers/queries, batched tool calls, no duplicate write-up on single-task plans, single-agent streaming, speculative retrieval, slimmer image); student picks A–D. No rule-based classifier, no semantic cache | Simple-question warm TTFT ≤ ~4–5 s (median of ≥ 5), cold TTFT ≤ 25 s after a real scale-to-zero, multi-agent questions reported separately, no quality/trace regressions | not started |
| **9. Evaluation** | Real numbers | Frozen test set (student-reviewed), RAGAS (4 metrics), provider comparison, cold and warm latency + TTFT, cost estimates | Results saved in `eval/results/` and reported in `evaluation.md` | not started |
| **10. Docs + submit** | Finish deliverables | Final `architecture.md`, `evaluation.md`, README with deploy steps and dated prices, `PROCESS_LOG.md` (student writes reflection), secret scan, zip inspection | Every row in `compliance.md` verified or justified; URL and zip ready | not started |

## Scope rules

- **Stretch goal:** the Code agent. If time runs short, drop it (three workers meets the floor).
- **Never dropped:** RAGAS, either provider, Tavily, MCP tool sharing, deployment, the required docs.
- **State:** stateless, single-turn throughout.

## Done so far (before Phase 0)

- Design proposal written (`design_proposal.md`).
- Scaffolding: `requirements.in`/`requirements.txt` (148 exact pins after the Phase 0 ragas fix, resolved for Python 3.13 Linux), `.env.example`, `.gitignore`, `.dockerignore`.
- Draft docs: `architecture.md`, `compliance.md`, `evaluation.md`, `README.md`, `dev_notes.md`.
- Course reference notes moved to `reference/` (git-ignored).
