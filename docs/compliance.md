# Requirement-to-Evidence Checklist

Working aid, not a graded deliverable. Source of truth: `Assignment_2_Description.pdf`.
Status values: **pending**, **implemented** (not yet verified), **verified** (evidence recorded), **blocked**, **n/a** (with justification).

Last updated: 2026-10-27 (Phase 2 first slice built; tester PHASE PASS).

## Orchestration (25%)

| Requirement | Where | How verified | Evidence | Status |
|---|---|---|---|---|
| LangGraph state graph compiles and runs end to end | `app/graph.py` | Integration test + deployed query | Local run 2026-10-27: planner → supervisor → research → supervisor → synthesize, streamed SSE (Self-RAG query, greeting, out-of-corpus query) | implemented |
| Supervisor pattern | `app/graph.py` | Trace shows supervisor ⇄ workers | `transition` events supervisor → research → supervisor → synthesize | implemented |
| ≥3 specialized workers (supervisor not counted) | `app/agents/` | Research, Summary, Data each invoked across test queries | | pending |
| Distinct system prompt per worker | `app/agents/prompts.py` | Code review | | pending |
| Distinct tool access per worker | `app/agents/*` | Tool lists differ; trace | | pending |
| Delegation driven by inspectable state fields | `route_next` | `transition` events carry reasons | Pure function of task status, dependencies, attempts, step count and time budget; unit-tested in `tests/unit/test_phase2_core.py` | implemented |
| Worker failure does not crash the graph | worker wrapper | Fault-injection test + partial answer | | pending |
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
| Out-of-corpus question → abstain or escalate | Research + synth prompts | System test | AlphaFold 3 query: "The available evidence does not answer this…", no invented sources (local, 2026-10-27) | implemented |
| RAGAS: faithfulness, answer relevance, context precision, context recall | `scripts/run_ragas.py` | Numeric results | | pending |

## LLM routing (15%)

| Requirement | Where | How verified | Evidence | Status |
|---|---|---|---|---|
| 2 different providers wired (OpenAI direct + Gemini via OpenRouter) | `app/llm/router.py` | Real calls to both | | pending |
| Both selected under ordinary queries | router | Provider share over eval set | | pending |
| Decision visible in UI/trace with rationale | `llm_route` event, Routing tab | Screenshot / SSE log | | pending |
| Rationale documented (hybrid) | `docs/architecture.md` §6 | | | pending |
| Exact verified model IDs in source + `.env.example` | `.env.example` | API check | `gpt-6.1-sol` (OpenAI /v1/models) and `google/gemini-3.8-flash` (OpenRouter catalog) both answered real streaming + tool calls 2026-10-27; `docs/testing/phase0_service_check.json` | implemented |
| Fallback traced separately | `fallback` event | Fault test | | pending |

## Frontend (15%)

| Requirement | Where | How verified | Evidence | Status |
|---|---|---|---|---|
| FastAPI SSE token streaming (incremental) | `app/main.py` | First token before `done` | Synth tokens forwarded from `astream`; Self-RAG run: 305 `token` events, first at about 29 s, `done` at 34.8 s | implemented |
| Vanilla HTML/JS served by FastAPI, no build | `app/static/` | | `index.html`, `app.js` (fetch + incremental SSE parser), `styles.css` | implemented |
| Visible routing decision per turn | Routing tab | | Routing tab lists every call (agent, provider, model, rationale, tokens); single default provider until Phase 4 | implemented (one provider) |
| Inspectable trace of orchestration | Trace tab | | Plan, transitions, tool calls, retrievals, worker results, errors, done | implemented |
| Citations linked to source identifiers | Sources tab | | `[C#]` markers become links only when in the registry; source viewer fetches the chunk and links the arXiv section | implemented |

## Deployment (15%)

| Requirement | Where | How verified | Evidence | Status |
|---|---|---|---|---|
| Public Cloud Run URL | README | Browser check | https://paper-analyst-g7b7dxmyuq-uw.a.run.app deployed 2026-10-27; corpus question answered after a measured cold start (17.9 s health, then 307 tokens / 5 citations); browser check pending (tester) | implemented |
| Arbitrary reasonable queries | | Unrehearsed queries | | pending |
| Cold start without failure | `scripts/probe_remote.py` | First request after the instance logged shutdown | 2026-10-27 07:39 UTC: health 17.9 s, then a corpus answer (TTFT 27.8 s, total 33.3 s); Cloud Run startup_latencies 17.6 s. One sample; more in Phase 9 | implemented |
| Credentials outside source | `scripts/env_to_yaml.py` temp env-vars file outside the repo | Secret scan | Image: no `.env*`; Cloud Build upload is an allowlist of 31 files; deploy log has no key values; temp dir deleted (2026-10-27) | implemented |
| Multi-stage python:3.13-slim, non-root, `$PORT` | `Dockerfile` | Container run as uid 10001 | Local 2026-10-27: `id` = 10001; PORT=9090 honored, default 8080; real query against the baked index as uid 10001 (non-owner uid fails: P2-N6 fix verified) | implemented |
| `==` pins in requirements.txt | `requirements.txt` | grep for `>=`/`~=` | 148 packages, all `==`; installs on py3.13.16 and all key imports incl. ragas succeed (2026-10-27) | implemented |
| Fresh-project deploy docs | README, `scripts/deploy.sh` | Dry run + real deploy | README *Deploy to a fresh GCP project*; script used for the first deploy 2026-10-27 (into an existing project, so the enable-API step was not exercised) | implemented |
| Durable state | | | Stateless single-turn; no persistence claimed | n/a |

## MCP tool sharing (10%)

| Requirement | Where | How verified | Evidence | Status |
|---|---|---|---|---|
| Tavily Remote MCP, Streamable HTTP, key as query param | `app/mcp_client.py` | `list_tools` + call | Phase 0 spike: list_tools + tavily_search OK 2026-10-27 (app client not built yet) | pending |
| Same tool used by Research and Summary in different contexts | traces | Deployed SSE log | | pending |
| Sharing rationale documented | `docs/architecture.md` §8 | | | pending |
| Key redacted from logs/traces | | grep logs for `tavilyApiKey=` | | pending |

## Deliverables

| Deliverable | Status |
|---|---|
| Live URL | pending |
| Source ZIP (no `.env`, inspected with `unzip -l`) | pending |
| `docs/architecture.md` | draft |
| `docs/evaluation.md` | outline only |
| `PROCESS_LOG.md` (200–300 words, student's own reflection) | pending |
| `README.md` | draft |
