# Requirement-to-Evidence Checklist

Working aid, not a graded deliverable. Source of truth: `Assignment_2_Description.pdf`.
Status values: **pending**, **implemented** (not yet verified), **verified** (evidence recorded), **blocked**, **n/a** (with justification).

Last updated: 2026-10-27 (Phase 1 corpus built; no application code yet).

## Orchestration (25%)

| Requirement | Where | How verified | Evidence | Status |
|---|---|---|---|---|
| LangGraph state graph compiles and runs end to end | `app/graph.py` | Integration test + deployed query | | pending |
| Supervisor pattern | `app/graph.py` | Trace shows supervisor ⇄ workers | | pending |
| ≥3 specialized workers (supervisor not counted) | `app/agents/` | Research, Summary, Data each invoked across test queries | | pending |
| Distinct system prompt per worker | `app/agents/prompts.py` | Code review | | pending |
| Distinct tool access per worker | `app/agents/*` | Tool lists differ; trace | | pending |
| Delegation driven by inspectable state fields | `route_next` | `transition` events carry reasons | | pending |
| Worker failure does not crash the graph | worker wrapper | Fault-injection test + partial answer | | pending |
| State schema documented | `docs/architecture.md` §4 | Matches code | | pending |

## RAG (20%)

| Requirement | Where | How verified | Evidence | Status |
|---|---|---|---|---|
| Substantial corpus (not a toy) | `data/corpus_seed.json`, `data/corpus_manifest.json` | Paper and chunk counts | 89 arXiv papers (88 full text, 1 metadata only), 912k words, 4,521 chunks, 1.62M tokens; 9 topic groups, 2022-12 to 2026-08 (2026-10-27) | verified |
| Ingestion → chunks → embeddings | `scripts/ingest.py`, `ingest/` | Ingest log | LaTeXML HTML → section records → 500/75-token section-bounded chunks → `text-embedding-3-small` (1536 d); first build embedded 4,523 chunks, $0.032; total spend about $0.035 | verified |
| Stored in Chroma | `data/chroma_index` (collection `arxiv_agents`, cosine) | Collection count | count 4,521 = manifest chunk total = Σ DB n_chunks (tester-verified); ids unique; `scripts/query_index.py` returns relevant sections; paraphrase queries retrieve the right paper | verified |
| Re-indexing adds no duplicates | `ingest/store.py` `sync_paper_chunks` | Second run adds 0 | Run 2: 0 added/0 updated/4,523 unchanged. After a parser change: 254 updated, 2 added, 2 deleted, count unchanged; next run 0 changes. After the acknowledgment fix: 4 updated, 2 deleted (4,521); next run 0 changes (2026-10-27). Tester Phase 1: PHASE PASS | verified |
| Top-k vector retrieval, k configurable | `app/rag.py` | Unit test | | pending |
| LCEL pipeline | `app/rag.py` | Code review | | pending |
| Augmentation prompt with chunk IDs | `app/rag.py` | Prompt in architecture.md matches code | | pending |
| Citations resolve to source chunks | `/api/source/chunk` | Resolution-rate check | | pending |
| Out-of-corpus question → abstain or escalate | | System test | | pending |
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
| FastAPI SSE token streaming (incremental) | `app/main.py` | First token before `done` | | pending |
| Vanilla HTML/JS served by FastAPI, no build | `app/static/` | | | pending |
| Visible routing decision per turn | Routing tab | | | pending |
| Inspectable trace of orchestration | Trace tab | | | pending |
| Citations linked to source identifiers | Sources tab | | | pending |

## Deployment (15%)

| Requirement | Where | How verified | Evidence | Status |
|---|---|---|---|---|
| Public Cloud Run URL | README | Browser check | | pending |
| Arbitrary reasonable queries | | Unrehearsed queries | | pending |
| Cold start without failure | | First request after idle | | pending |
| Credentials outside source | env vars file | Secret scan | | pending |
| Multi-stage python:3.13-slim, non-root, `$PORT` | `Dockerfile` | Container run as uid 10001 | | pending |
| `==` pins in requirements.txt | `requirements.txt` | grep for `>=`/`~=` | 148 packages, all `==`; installs on py3.13.16 and all key imports incl. ragas succeed (2026-10-27) | implemented |
| Fresh-project deploy docs | README | | | pending |
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
