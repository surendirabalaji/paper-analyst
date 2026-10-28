# Requirement-to-Evidence Checklist

Working aid, not a graded deliverable. Source of truth: `Assignment_2_Description.pdf`.
Status values: **pending**, **implemented** (not yet verified), **verified** (evidence recorded), **blocked**, **n/a** (with justification).

Last updated: 2026-10-27 (Phase 6 done: Data agent, hand-offs, D# citations; tester PHASE PASS on rev 00010).

## Orchestration (25%)

| Requirement | Where | How verified | Evidence | Status |
|---|---|---|---|---|
| LangGraph state graph compiles and runs end to end | `app/graph.py` | Integration test + deployed query | Local run 2026-10-27: planner → supervisor → research → supervisor → synthesize, streamed SSE (Self-RAG query, greeting, out-of-corpus query) | implemented |
| Supervisor pattern | `app/graph.py` | Trace shows supervisor ⇄ workers | `transition` events supervisor → research → supervisor → synthesize | implemented |
| ≥3 specialized workers (supervisor not counted) | `app/agents/research.py`, `summary.py`, `data.py` | Research, Summary, Data each invoked across test queries | Phase 6 local (2026-10-27): the real planner chose data, data → summary, data → research, summary and research across 8 questions (`eval/results/phase6/local/planner_probe.txt`); Data → Summary ran end to end (`eval/results/phase6/local/`, `live_test_handoff.events.jsonl`). Code agent not built (decision in architecture §5). Deployed on rev 00010: Data → Summary hand-off verified by the tester (`docs/testing/phase-6-retest2-20261027T2230Z.md`) | verified (3/3) |
| Distinct system prompt per worker | `app/agents/prompts.py` | Code review | `RESEARCH_ROLE` (cross-paper, corpus first, web for current developments), `SUMMARY_ROLE` (one paper at a time, section by section, web only for status, works on handed-off papers), `DATA_ROLE` (schema first, one read-only SELECT, repair once, empty = answer, question text untrusted) | implemented (3/3) |
| Distinct tool access per worker | `app/agents/*` | Tool lists differ; trace | Research: `corpus_search` + shared `tavily_search`; Summary: `get_paper_outline`, `get_paper_section`, `search_within_paper` + shared `tavily_search`; Data: `get_schema`, `run_sql` only (no web, no vector search). Unit tests assert the bound tool sets | implemented (3/3) |
| Delegation driven by inspectable state fields | `route_next` | `transition` events carry reasons | Pure function of task status, dependencies, attempts, step count, time budget and upstream results (a Summary whose Data dependencies found no papers is skipped as `no_upstream_papers`); hand-offs carry Data's paper ids in `task.upstream_papers` and are traced as `handoff` events. Unit-tested in `tests/unit/test_phase2_core.py` and `tests/unit/test_phase6_data.py` | implemented |
| Worker failure does not crash the graph | worker wrapper (`make_worker_node`), `app/faults.py`, `route_next` retry/skip rules, `synthesize` gaps + `failure_answer` | Fault-injection test + partial answer | Phase 7: 41 offline tests in `tests/unit/test_phase7_failures.py` (timeout, hang, exception, empty, tool error, both providers down, budget exhaustion, dependency failure, partial dependency, retry then success, guard 403, P5-N1/P6-D3/P6-D1c/N9); live local test `tests/integration/test_phase7_live_fault.py` (real providers, injected content-worker failure → `partial`, D# answer + "Incomplete evidence" note, `eval/results/phase7/`). Deployed rev 00011: forced worker failures give honest partial answers with a visible trace; guard 403 without token; tester PHASE PASS (`docs/testing/phase-7-20261028T0100Z.md`) | verified |
| State schema documented | `docs/architecture.md` §4 | Matches code | | pending |

## RAG (20%)

| Requirement | Where | How verified | Evidence | Status |
|---|---|---|---|---|
| Substantial corpus (not a toy) | `data/corpus_seed.json`, `data/corpus_manifest.json` | Paper and chunk counts | 89 arXiv papers (88 full text, 1 metadata only), 912k words, 4,521 chunks, 1.62M tokens; 9 topic groups, 2022-12 to 2026-08 (2026-10-27) | verified |
| Ingestion → chunks → embeddings | `scripts/ingest.py`, `ingest/` | Ingest log | LaTeXML HTML → section records → 500/75-token section-bounded chunks → `text-embedding-3-small` (1536 d); first build embedded 4,523 chunks, $0.032; total spend about $0.035 | verified |
| Stored in Chroma | `data/chroma_index` (collection `arxiv_agents`, cosine) | Collection count | count 4,521 = manifest chunk total = Σ DB n_chunks (tester-verified); ids unique; `scripts/query_index.py` returns relevant sections; paraphrase queries retrieve the right paper | verified |
| Re-indexing adds no duplicates | `ingest/store.py` `sync_paper_chunks` | Second run adds 0 | Run 2: 0 added/0 updated/4,523 unchanged. After a parser change: 254 updated, 2 added, 2 deleted, count unchanged; next run 0 changes. After the acknowledgment fix: 4 updated, 2 deleted (4,521); next run 0 changes (2026-10-27). Tester Phase 1: PHASE PASS | verified |
| Top-k vector retrieval, k configurable | `app/rag.py` | Unit test | `Retriever.search`, k clamped 3–10, `RAG_TOP_K` default 6; `retrieval` events show k and scores | implemented |
| LCEL pipeline | `app/rag.py` | Code review | `make_corpus_search_chain` (retrieve \| register) and `AUGMENT_PROMPT \| chat_model` streaming synthesis | implemented |
| Augmentation prompt with chunk IDs | `app/rag.py` | Prompt in architecture.md matches code | `AUGMENT_SYSTEM`/`AUGMENT_HUMAN`; evidence items carry cite id and chunk_id | implemented |
| Citations resolve to source chunks | `/api/source/chunk` | Resolution-rate check | Self-RAG run: 5/5 cited markers resolved, 0 unresolved; unknown or malformed ids give 404 | implemented |
| Out-of-corpus question → abstain or escalate | Planner `needs_live`, Research web gate (`RAG_MIN_SCORE`), synth prompt | System test | AlphaFold 3 query: abstained, no invented sources (Phase 2). Phase 5 (P2-N8): off-topic questions now escalate to live web search: "capital of Australia" → Research `needs_live` → 4 `W#` web citations; "MCP authorization spec" (best corpus score 0.52) → corpus + web. Abstentions no longer attach unrelated chunks (P3-N3): synthesis gets only evidence a worker cited, and markers attached only to "evidence does not answer" clauses are not linked (`done.dropped_citations`) | implemented |
| RAGAS: faithfulness, answer relevance, context precision, context recall | `scripts/run_ragas.py` | Numeric results | | pending |

## LLM routing (15%)

| Requirement | Where | How verified | Evidence | Status |
|---|---|---|---|---|
| 2 different providers wired (OpenAI direct + Gemini via OpenRouter) | `app/llm/router.py` (`Router.client`, `OpenRouterChat`) | Real calls to both | `tests/integration/test_phase4_live_routing.py` passed 2026-10-27 (both providers status ok; Gemini upstream "Google AI Studio" recorded per call); `eval/results/phase4/after/` | implemented |
| Both selected under ordinary queries | `app/llm/policy.py` | Two ordinary questions; provider share over eval set in Phase 9 | 2026-10-27 local: Self-RAG question synthesized on Gemini (cost, 1 paper); Reflexion vs Self-Refine on OpenAI (capability, comparison over 2 papers); live test asserts this | implemented (deployed 2026-10-27: Reflexion vs Self-Refine and tree-search questions on OpenAI, P3-N2 question on Gemini; `eval/results/phase4/deployed/`) |
| Decision visible in UI/trace with rationale | `llm_route` event, Routing tab, per-answer routing line | SSE log + browser | `llm_route` carries agent, call site, provider, exact model ID, decision inputs, dimension, rationale; browser check by tester pending | implemented |
| Rationale documented (hybrid) | `docs/architecture.md` §6 | Review | Routing table, features, thresholds, prices, fallback rules | implemented |
| Exact verified model IDs in source + `.env.example` | `.env.example`, `app/config.py`, `app/llm/policy.py` PRICES | API check | `gpt-6.1-sol` (OpenAI /v1/models) and `google/gemini-3.8-flash` (OpenRouter catalog) both answered real streaming + tool calls 2026-10-27; re-verified with prices 2026-10-27 (Phase 4): OpenAI pricing page $2/$10, OpenRouter catalog $0.825/$4.125 | implemented |
| Fallback traced separately | `fallback` event (`original_decision_id`) | Fault test | Unit tests (retry, then fallback; auth error, direct fallback; stream falls back only before first token; cancel still emits `llm_result`); live test: invalid Gemini model ID gives a real OpenRouter error, then OpenAI ok (2026-10-27) | implemented |

## Frontend (15%)

| Requirement | Where | How verified | Evidence | Status |
|---|---|---|---|---|
| FastAPI SSE token streaming (incremental) | `app/main.py`, `app/static/app.js` | Browser MutationObserver: answer length changes before `done` | Phase 8 headless Chrome on rev 00013 (2026-10-28), `tests/e2e/phase8_browser_ui.py`, 20/20 checks, `eval/results/phase8/`: 69 answer-length changes before the `done` summary (first token → done 3.7 s); streaming cursor seen. Unicode question (French + Chinese + emoji) streamed intact with no U+FFFD | verified |
| Vanilla HTML/JS served by FastAPI, no build | `app/static/` (`index.html`, `app.js`, `md.js`, `styles.css`) | Unit tests | `tests/unit/test_phase8_ui.py`: only `/static/md.js` and `/static/app.js`, no external assets, no `node_modules`, no `innerHTML`/`eval`/`new Function`; Markdown parser checked under Node (`tests/unit/js/md_check.mjs`: injection stays text, Unicode, partial input, adversarial input) | verified |
| Visible routing decision per turn | Per-turn routing chips + Routing tab | Browser | Phase 8 headless Chrome on rev 00013 (2026-10-28), `tests/e2e/phase8_browser_ui.py`, 20/20 checks, `eval/results/phase8/`: chips "planner → Gemini (cost) · data → Gemini ×3 (cost) · summary → Gemini ×3 (cost) · synthesizer → OpenAI (capability)"; Routing tab rows with provider, exact model, dimension, rationale and inputs for both providers (`03_routing_tab.png`) | verified |
| Inspectable trace of orchestration | Trace tab (grouped per turn and per agent) | Browser | Phase 8 headless Chrome on rev 00013 (2026-10-28), `tests/e2e/phase8_browser_ui.py`, 20/20 checks, `eval/results/phase8/`: collapsible turn → agent groups (Orchestration, Data, Summary, Synthesis; Research + MCP rows on the web turn) with done/partial/failed/skipped/injected badges, `+t` times and sentence transitions (`04_trace_grouped.png`, `09_trace_unicode_turn_mcp.png`) | verified |
| Citations linked to source identifiers | Answer links + Sources tab + viewer | Browser | Phase 8 headless Chrome on rev 00013 (2026-10-28), `tests/e2e/phase8_browser_ui.py`, 20/20 checks, `eval/results/phase8/`: C1 → viewer with the paper title and section (not the chunk id), full chunk, arXiv section and abstract links; D1 → SQL and rows; W1 → full snippet and page link (`05`, `06`, `08` screenshots) | verified |
| Error states and disconnects never hang | `app.js` `fail()`, stall watchdog | Browser | Phase 8 headless Chrome on rev 00013 (2026-10-28), `tests/e2e/phase8_browser_ui.py`, 20/20 checks, `eval/results/phase8/`: a 403 (wrong fault token) gives "Error: the server refused the request (HTTP 403): …" with a failed badge and no "Planning…" (`01_http_error_state.png`); the token is cleared from the URL. A 60 s no-byte watchdog and a closed-before-`done` path are in code | verified (HTTP error); implemented (stall) |
| Dark-mode readability | `styles.css` | Browser contrast check | Viewer link and citation link contrast 9.75:1 on the dark panel (`10_dark_mode_viewer.png`) | verified |

## Deployment (15%)

| Requirement | Where | How verified | Evidence | Status |
|---|---|---|---|---|
| Public Cloud Run URL | README | Browser check | https://paper-analyst-g7b7dxmyuq-uw.a.run.app deployed 2026-10-27; corpus question answered after a measured cold start (17.9 s health, then 307 tokens / 5 citations); browser check pending (tester) | implemented |
| Arbitrary reasonable queries | | Unrehearsed queries | | pending |
| Cold start without failure | `scripts/probe_remote.py` | First request after the instance logged shutdown | 2026-10-27 07:39 UTC: health 17.9 s, then a corpus answer (TTFT 27.8 s, total 33.3 s); Cloud Run startup_latencies 17.6 s. One sample; more in Phase 9 | implemented |
| Credentials outside source | `scripts/env_to_yaml.py` temp env-vars file outside the repo | Secret scan | Image: no `.env*`; Cloud Build upload is an allowlist of 31 files; deploy log has no key values; temp dir deleted (2026-10-27). P3-N1 (keys are plain Cloud Run env vars, readable via `gcloud run services describe` and in deploy audit logs) **accepted by the student 2026-10-27**; no Secret Manager | implemented (P3-N1 accepted) |
| Multi-stage python:3.13-slim, non-root, `$PORT` | `Dockerfile` | Container run as uid 10001 | Local 2026-10-27: `id` = 10001; PORT=9090 honored, default 8080; real query against the baked index as uid 10001 (non-owner uid fails: P2-N6 fix verified) | implemented |
| `==` pins in requirements.txt | `requirements.txt` | grep for `>=`/`~=` | 148 packages, all `==`; installs on py3.13.16 and all key imports incl. ragas succeed (2026-10-27) | implemented |
| Fresh-project deploy docs | README, `scripts/deploy.sh` | Dry run + real deploy | README *Deploy to a fresh GCP project*; script used for the first deploy 2026-10-27 (into an existing project, so the enable-API step was not exercised) | implemented |
| Durable state | | | Stateless single-turn; no persistence claimed | n/a |

## MCP tool sharing (10%)

| Requirement | Where | How verified | Evidence | Status |
|---|---|---|---|---|
| Tavily Remote MCP, Streamable HTTP, key as query param | `app/mcp_tavily.py` (`TavilyMCP`) | `list_tools` + real searches | `langchain-mcp-adapters` `MultiServerMCPClient`, transport `streamable_http`, URL `https://mcp.tavily.com/mcp/?tavilyApiKey=…`; unit test asserts the connection has only `transport` and `url`. Live test 2026-10-27: real search parsed with URLs | implemented |
| Live (current) retrieval distinct from static RAG | Research (`needs_live` or low corpus support) and Summary (status) | Current-information query | Local 2026-10-27: "latest developments in LLM agent memory this year" → web results from 2026 cited as `W#` with retrieval time. Deployed (rev 00005-zc8): "agent frameworks or SDK releases in the past month" → `time_range=month`, 6 `W#` sources dated October 2026 (`eval/results/phase5/deployed/2.events.jsonl`) | implemented (deployed) |
| Live retrieval failures handled | `TavilyUnavailable` kinds: auth, quota, rate_limit, connection, timeout, empty, budget, disabled | Unit + live fault | Fake key against the real server (local, real providers): 401 → `auth`, Research `partial` / `live_search_unavailable`, corpus-only answer that says web results were unavailable; key absent from debug-level server log and SSE | implemented |
| Same tool used by Research and Summary in different contexts | `shared_search_tool` (`app/mcp_tavily.py`), traces | Live test + deployed SSE log | Live test 2026-10-27 (`tests/integration/test_phase5_live_mcp_sharing.py`, `eval/results/phase5/live_test_sharing.events.jsonl`): `web_search` events from `research` ("find current developments on the topic") and `summary` ("check the current status of a specific paper") on `tavily-remote-mcp`. Deployed (rev 00005-zc8, `eval/results/phase5/deployed/1.events.jsonl`): `research` 1 call and `summary` 2 calls to `tavily_search`, with distinct purposes and queries. Tester verified independently on rev 00005 (SSE + browser) and rev 00006 (browser) | verified (Phase 5 PASS) |
| Sharing rationale documented | `docs/architecture.md` §8 | Review | §8 written in Phase 5 | implemented |
| Key redacted from logs/traces | `app/events.py` (`redact`, `safe_error`, `RedactingFilter`), logger levels in `app/main.py` | Fake-key tests | Unit tests: httpx status error with the key in the URL, ExceptionGroup, URL-encoded key, log args and tracebacks, real MCP client connection failure (error, logs at DEBUG, SSE frames). Local server with `--log-level debug`: 0 occurrences of the fake key or `tavilyApiKey`. Cloud Run (388 entries since the deploy): 0 in application logs; the key appears only in the audit-log `ReplaceService` env vars (P3-N1, accepted) | implemented |

## Deliverables

| Deliverable | Status |
|---|---|
| Live URL | pending |
| Source ZIP (no `.env`, inspected with `unzip -l`) | pending |
| `docs/architecture.md` | draft |
| `docs/evaluation.md` | outline only |
| `PROCESS_LOG.md` (200–300 words, student's own reflection) | pending |
| `README.md` | draft |
