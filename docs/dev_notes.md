# Development Notes

Factual log of what happened during development. It is the source for `PROCESS_LOG.md` (200–300 words). Record facts as they happen: what worked, what failed, what you had to decide yourself.

## 2026-10-26

- **AI tools:** Claude Code (Claude Opus 5.5, model ID `claude-opus-5-5`) for reading the assignment, design and scaffolding; a `research-architect` subagent drafted the architecture proposal (`docs/design_proposal.md`).
- **Ideas considered:** Dev Docs Copilot, text-to-SQL (World Bank WDI; Chinook + SQLite docs; F1), arXiv Paper Analyst. Chose the **Agentic AI Paper Analyst**.
- **Provider decision:** OpenAI (direct key) plus Gemini through OpenRouter as the second provider.
- **Design decisions:**
  - arXiv HTML instead of PDF parsing.
  - Code agent works over pinned repo snapshots, not live cloning.
  - Deterministic supervisor after an LLM planner.
  - Tavily `tavily_search` shared by Research and Summary.
  - Stateless single-turn.
- **Found during design:** Assignment 1 sent the Tavily key as a Bearer header; Assignment 2 requires it as the `tavilyApiKey` URL query parameter.
- **Dependency check:**
  - `mcp` 2.x conflicts with `langchain-mcp-adapters` 0.3.2 (requires mcp <2), so we pinned `mcp==1.30.0`.
  - The full set (144 packages, including ragas 0.4.3) resolves for Python 3.13 on Linux.
- **Local environment:** Python 3.14 installed by default; Python 3.13.16 installed via uv for this project.

### Phase 0 (in progress)

- **Git:** repo initialised and pushed to a private GitHub repo (`surendirabalaji/paper-analyst`); private because the assignment forbids sharing source between students. The assignment PDF is git-ignored (instructor material, kept locally).
- **Install on 3.13:** all pinned packages install into a Python 3.13.16 venv and `uv pip check` passes.
- **Bug found: ragas import failure.** `import ragas` (0.4.3, latest) failed: it imports `langchain_community.chat_models.vertexai`, which `langchain-community` 0.4.2 removed. A clean install did not catch this; only an import test did. Fix: hold `langchain-community==0.4.1` (still has the module, accepts langchain-core 1.x). Lockfile recompiled: 148 exact pins. All four RAGAS metrics import. Those legacy class names are deprecated in 0.4.3; use `ragas.metrics.collections` `Faithfulness`, `AnswerRelevancy`, `ContextPrecisionWithReference`, `ContextRecall` (tester finding R5).
- **Model IDs (OpenRouter public catalog, 2026-10-26):** `google/gemini-3.8-flash` listed, tools supported, $0.825 in / $4.125 out per 1M tokens. `openai/gpt-6.1-sol` listed at $2 / $10 (OpenRouter price; OpenAI direct price still to confirm). Cost ordering assumed by the router holds (Gemini cheaper).
- **arXiv HTML samples** (2210.03629 ReAct, 2310.11511 Self-RAG, 2401.15884 CRAG): all three return HTML, including a 2022 paper. LaTeXML gives stable IDs (`S3`, `S3.SS2`, appendix `A1.SS1`), `<math alttext>`, figcaptions, and a separate `ltx_bibliography`. 7.6k–18.7k words excluding the bibliography. Parser issues for Phase 1: `get_text(" ")` fragments small-caps headings ("Re asoning"); the HTML URL has no version, so take the version from the arXiv API.
- `scripts/phase0_check.py` run 2026-10-27 03:50 UTC, all three OK (evidence: `docs/testing/phase0_service_check.json`):
  - **OpenAI direct `gpt-6.1-sol`:** listed in `/v1/models`, streamed 34 content chunks, TTFT 2.1 s, usage returned, tool call well formed. `text-embedding-3-small` returns 1536 dims.
  - **Gemini `google/gemini-3.8-flash` via OpenRouter:** streaming, usage and tool calling work. But a one-sentence answer used 412 reasoning tokens of 455 output, TTFT 5.6 s, and arrived in only 3 chunks. Reasoning effort must be controlled (Phase 4) for cost and smooth token streaming.
  - **Tavily Remote MCP:** Streamable HTTP with `tavilyApiKey` query param works via langchain-mcp-adapters. Tools: `tavily_search`, `tavily_extract`, `tavily_crawl`, `tavily_map`, `tavily_research`. One search returned 3 results in 2.7 s.
  - Secret scan: no key value in any repo file.
- **Tester Phase 0 gate: PHASE PASS** (`docs/testing/phase-0-20261027T0400Z.md`). Defect P0-D1 (Medium): Tavily returns auth failures as a normal tool result (`{"error":..., "status":401}`), and listing tools works even with a bad key, so the check script reported `ok` with an invalid key. Fixed: the script now parses the result and fails on an error field, status ≥ 400 or zero results; redaction also covers `tavilyApiKey%3D`. The same parsing is required in the Phase 5 MCP client so error JSON is never cited as web evidence.
- Risks carried forward: Gemini reasoning-token overhead and chunky streaming (Phase 4); record OpenRouter upstream/snapshot per call (Phase 4); Tavily results lack `published_date`, so use retrieval time (Phase 5); check arXiv's 2026-10-01 rate-limit policy before bulk fetch (Phase 1).
- **Tester retest: PHASE PASS** (`docs/testing/phase-0-retest-20261027T0400Z.md`). P0-D1 confirmed fixed against the real service (a fake key now gives `error`; a real key still gives `ok`). New low note P0-I4 for Phase 5: the MCP client should count a search as successful only when it returns parsed results with URLs, not merely when no known error shape is found. Phase 0 marked done.

### Phase 1: Corpus (branch `phase-1-corpus`)

- **Scope decision (student, 2026-10-26):** no GitHub repo snapshots in Phase 1. Only repo links are recorded (`repos` table). Snapshots move to Phase 6, together with the optional Code agent.
- **arXiv access (checked 2026-10-26):** the API terms still say at most 1 request every 3 s on one connection. The 2026-10-01 change caps *submissions* (2 papers a month per submitter), and third-party write-ups (not arXiv itself) mention stricter, unpublished bulk-access limits. In practice the API answered our first requests with `429 Rate exceeded` (no Retry-After), even at low volume. Fix: one throttled connection (3.5 s gap) with backoff of 20/45/90/180 s on 429/5xx, plus caching of metadata and HTML in `data/raw/`. A cold fetch of 89 papers took about 9 minutes (91 requests).
- **Embedding price:** `text-embedding-3-small` costs $0.02 per 1M tokens (OpenAI model page, checked 2026-10-26).
- **Corpus selection:** 59 well-known foundational IDs proposed by Claude Code, all checked against arXiv API titles (59/59 matched). Then 7 title searches limited to 2025-01..2026-09; I chose 30 on-topic papers from the results. All 89 have HTML. 88 pass the 3,000-word rule; 2506.16988 has 2,721 words and is metadata only.
- **Parser bugs found and fixed against real pages:**
  - Self-RAG's reflection tokens are inline TikZ SVGs. Dropping SVGs lost them ("if == Yes"); now short foreignObject labels are kept ("if Retrieve == Yes").
  - Math `alttext` sometimes holds raw pgf source (`\lxSVG@...`) when a token box sits inside an equation. Self-RAG went from 15k words to 88k tokens (12.7 tokens per word in one appendix). Now pgf-looking or >400-character alttext is replaced by the drawn labels. Self-RAG: 221 chunks down to 54.
  - Small-caps headings are kept whole by joining inline nodes without separators.
  - Headings get breadcrumbs, so subsections of sections with no prose of their own keep their parent title.
  - Acknowledgments are dropped.
  - The "⬇" download link on code listings (`.ltx_listing_data`) was in 256 chunks; it is now dropped.
- **Repo discovery:** 58 of 89 papers. Order checked: arXiv comment, abstract, HTML front matter, then introduction links only when release wording is near. All 58 were checked by eye as the paper's own repo. Self-RAG correctly has none, because it links a `github.io` page and its introduction cites third-party repos. Fixed a `.git.` suffix normalisation bug.
- **Build results (2026-10-27):**
  - Run 1: 4,523 chunks added, 1,619,349 tokens embedded, $0.0324, 83 s.
  - Run 2: 0 added, 0 updated, 0 deleted, 4,523 unchanged.
  - Run 3, after the ⬇ parser fix: 2 added, 254 updated, 2 deleted, count still 4,523, $0.0021.
  - Run 4: 0 changes.
  - Embedding total so far: about $0.035.
- **Counts (after the P1-D3 fix):** 89 papers, 88 full text, 911,513 words indexed, 4,521 chunks (4,523 before the fix), median 400 tokens (min 21, max 544 with prefix; 59 chunks under 60 tokens from very short sections). Chroma index 105 MB on disk, `papers.db` 316 KB.
- **Smoke test:** `scripts/query_index.py "how does Self-RAG decide when to retrieve?"` returns only Self-RAG chunks, top hits §3.3 and A.3, cosine relevance about 0.61–0.63. "why do multi-agent LLM systems fail?" returns MAST (2503.13657) chunks at about 0.77–0.80. Relevance scores vary by query, so `RAG_MIN_SCORE` must be calibrated in Phase 2.
- **Known limitations:** 3 papers predate the design's 2023-01 start (ReAct, IRCoT, HyDE, all 2022-12) and are kept as foundational. A few primary categories fall outside cs.CL/AI/LG/MA/IR (cs.SE, cs.HC, cs.OS, cs.CR, cs.CY). The seed records this. Some algorithm listings still have choppy line breaks; content is intact.
- **Tester Phase 1 gate: PHASE PASS** (`docs/testing/phase-1-20261027T0451Z.md`; 104 passed, 13 real-index checks passed). Four defects, all fixed:
  - **P1-D1 (Medium):** `--limit` rewrote `papers.db` with only N papers while the index kept all of them. It now skips the DB write.
  - **P1-D2 (Low):** a duplicate seed ID crashed after the index sync. Duplicates are now dropped with a warning before any work.
  - **P1-D3 (Low):** acknowledgments leaked in three forms: headings with extra words ("Acknowledgments and Disclosure of Funding"), `ltx_acknowledgements` blocks inside conclusions (3 papers), and a bold "Acknowledgement" pseudo-heading paragraph (MetaGPT). All three are now removed. The re-ingest updated 4 chunks and deleted 2 (now 4,521), and the next run made 0 changes. No "We thank" text remains in the corpus; the remaining "acknowledg-" matches are the ordinary verb in body text.
  - **P1-D4 (Low):** the dimensions were hard-coded at 1536, so a model change plus `--rebuild` could leave an empty, mislabelled collection. Now dimensions are measured from the model before any create or drop, and an existing collection uses its stored dimensions, so a no-op rerun still makes zero embedding calls. My first attempt probed on every run; the tester's "rerun makes no embed calls" test caught that. `query_index.py` now checks dimensions too.
- Embedding total after fixes: about $0.035.
- **Tester retest: PHASE PASS** (`docs/testing/phase-1-retest-20261027T0459Z.md`). D1–D4 confirmed fixed. 113 passed offline; 120 passed with live and real-index tests. A real rerun made 0 changes, and `papers.db` was byte-identical. Open notes: P1-N1 (equations with token boxes keep only their labels), P1-N2 (`papers.updated` holds the latest-version date, which equals the pinned version today), P1-N3 (pytest is not pinned in requirements). Also a residual edge case: if a model with the same name changed its dimensions, the Phase 2 startup check must catch it. Phase 1 marked done.

### Phase 2: First working version (branch `phase-2-first-slice`)

- **Built:** `app/` package: FastAPI app (`main.py`), SSE emitter (`events.py`), LangGraph graph (`graph.py`: planner → deterministic supervisor → Research → synthesize), generic bounded tool loop (`agents/worker.py`), Research agent with `corpus_search` (`agents/research.py`), LCEL retrieval chain, evidence registry and augmentation prompt (`rag.py`), router stub (`llm/router.py`), and the vanilla chat page (`static/`).
- **Decisions:**
  - Events go through a per-request `asyncio.Queue` instead of LangGraph stream modes. It is simpler and keeps event order exact. Tokens are pushed from the synthesizer's `astream` as they arrive.
  - Phase 2 sends every LLM call to `DEFAULT_PROVIDER=openai` (`gpt-6.1-sol`). Gemini streamed in large chunks with many reasoning tokens in Phase 0, so it waits for the Phase 4 router. The rationale shown in the UI says routing is not built yet.
  - The synthesizer gets only the evidence items the worker cited, not everything retrieved.
- **Local runs (2026-10-27, OpenAI only):**
  - "How does Self-RAG decide when to retrieve?": 1 Research task, 3 `corpus_search` calls, all Self-RAG chunks. 305 streamed tokens, 5/5 citations resolved. TTFT about 29 s, total 34.8 s, 19.2k input / 1.1k output tokens over 6 calls.
  - "hi there!": empty plan, short reply with an example question, no citations, 5.1 s.
  - AlphaFold 3 (not in the corpus): the agent abstained ("The available evidence does not answer this…"). The first run made 7 searches and used **68.6k input tokens**, because every search result stays in the loop context. Fix: a per-task cap of 4 tool calls (`MAX_CALLS_PER_TASK`). Rerun: 4 calls, 29.1k input tokens, TTFT 18 s.
- **Known issue for Phase 4:** latency is high. The planner takes about 5 s, the Research loop 15–23 s (3–4 rounds on `gpt-6.1-sol`), and the synthesizer TTFT about 1 s. Routing the planner and loop calls to the cheaper model, and a smaller loop, are the levers.
- **Checks:** 120 offline tests pass (6 new in `tests/unit/test_phase2_core.py`). `node --check` passes on `app.js`. Unknown, malformed and `../` chunk ids return 404. An empty query returns 422.
- **Tester Phase 2 gate: PHASE PASS** (`docs/testing/phase-2-20261027T0649Z.md`). It drove a real headless Chrome against the local app: tokens streamed, 7 citation links appeared, and the source viewer matched `/api/source/chunk`. 159 passed, 6 skipped, 1 xfailed (39 new tester tests). Live spend about $0.28. Open non-blocking notes: P2-N1 latency 10–28 s TTFT (Phase 4/9); P2-N2 the model picks k per call, so `RAG_TOP_K` is only a default; P2-N3 the source viewer title shows the chunk id; P2-N4 a failed LLM call leaves its route without a result event (Phase 4/7); P2-N5 raw Markdown and dark-mode link contrast (Phase 8); P2-N6 Chroma writes to `chroma.sqlite3` on open, a Phase 3 container risk; P2-N7 the 120 s request budget is not strictly enforced (Phase 7); P2-N8 out-of-domain questions get no retrieval (decide in Phase 5).

### Phase 3: Early deploy (branch `phase-3-early-deploy`)

- **Student decisions (2026-10-27):** reuse the existing GCP project `changelens-510807` (Assignment 1; billing on; Run, Cloud Build and Artifact Registry already enabled, so no API was enabled) with a new Artifact Registry repo and a new service, both named `paper-analyst`, in `us-west1`. The A1 `changelens` service was not touched. Live test budget for Phase 3: $1.
- **Built:** multi-stage `Dockerfile` (python:3.13-slim, `/opt/venv`, uid 10001, `uvicorn` on `${PORT:-8080}`); allowlist `.dockerignore` and `.gcloudignore`; `scripts/deploy.sh` and `scripts/env_to_yaml.py` (adapted from my A1 deploy script); `scripts/probe_remote.py` (client-side health/TTFT/total timing); a stdout log handler with startup timing in `app/main.py`.
- **P2-N6 (Chroma write on open) confirmed and fixed:** the same image run as uid 10002 fails at startup with `attempt to write a readonly database`; as the owner uid 10001 it opens the index and answers queries. Fix: `COPY --chown=10001:10001` for `data/chroma_index` only. Copying to `/tmp` was rejected (in-memory on Cloud Run, about 105 MB per cold start).
- **Found while building:** `ingest/chunking.py` calls `tiktoken.get_encoding()` at import time, so a fresh container would download the BPE file on every cold start. The builder stage now pre-fetches it into `/opt/tiktoken` (`TIKTOKEN_CACHE_DIR`).
- **Bug in my first `.gcloudignore`:** `!data/` re-included all of `data/raw/` (179 cached HTML/JSON files, 47 MB). Caught with `gcloud meta list-files-for-upload`; fixed with explicit re-excludes. The upload is now exactly 31 files.
- **Bug in my probe/smoke parser:** it read `message` from `error` events, but the field is `message_safe`, so errors looked empty. Fixed in `probe_remote.py` and `deploy.sh`.
- **Local container (2026-10-27, arm64 Docker Desktop):** image 1.86 GB (venv 1.2 GB, mostly ragas/datasets/pyarrow/scipy/kubernetes pulled by ragas and chromadb). `PORT=9090` honored; default 8080 without PORT. Startup (index open) 0.41 s. One Self-RAG query as uid 10001: 342 tokens, 6 citations, no errors. Memory 230 MiB idle, 245 MiB after a query. No `.env` in the image. Path traversal on `/api/source/chunk` returns 404.
- **First deploy (2026-10-27 07:22 UTC):** revision `paper-analyst-00001-b8n`, image `paper-analyst:20261027-071707-6f99342-dirty` (425 MB compressed in Artifact Registry). URL `https://paper-analyst-g7b7dxmyuq-uw.a.run.app`. Settings: 2 GiB, 1 CPU, startup CPU boost, concurrency 10, timeout 300 s, min 0, max 3. Smoke test passed (302 tokens, 6 citations, `done`). The script's own output contains no key values and the temp env-vars dir was deleted. But the keys are plain Cloud Run env vars: readable via `gcloud run services describe` and recorded in the Cloud Audit Logs `CreateService` entries (tester P3-N1).
- **Cold start (measured):** waited until the instance logged `Finished server process` (07:38:41, about 15 min idle), then at 07:39:47 the first `/health` took **17.9 s** client-side. Logs: uvicorn started 07:40:00 (≈13 s of container start and imports) and the index opened in 4.90 s. The Cloud Run `startup_latencies` metric: 17.6 s (and 16.3 s for the first deploy's instance). The Self-RAG question that followed: TTFT 27.8 s, total 33.3 s, 307 tokens, 5 citations. **Exit check met: public URL answered a corpus question after a cold start.** One sample only; Phase 9 will measure more.
- **Warm runs:** Self-RAG again TTFT 34.1 s / total 37.9 s; "Why do multi-agent LLM systems fail?" first run failed with a partial result after 97 s (2 tasks, 2 worker `error` events, 7 `llm_route` vs 5 `llm_result`; consistent with two 45 s worker timeouts) and still ended with `done`; a rerun succeeded in 59 s with 9 MAST citations. This is the known latency problem (P2-N1/N4/N7), not a container issue.
- **Memory on Cloud Run:** `container/memory/utilizations` 0.239–0.248 of 2 GiB (≈ 490–510 MiB) idle and through 4 sequential queries. Kept at 2 GiB for now: concurrent load is untested and Phases 5–6 add MCP sessions and the Data agent. Re-measure under concurrency before lowering (1 GiB looks possible).
- **Startup idea not done:** a runtime-only requirements set without ragas/datasets would shrink the image; import time would not change, since the app never imports ragas.
- **Tester Phase 3 gate: PHASE PASS** (`docs/testing/phase-3-20261027T0745Z.md`). Independent cold start (scale-to-zero confirmed in logs): the page loaded in headless Chrome after 35.1 s and answered an unrehearsed, indirectly worded Gorilla question with streamed tokens and 2 resolving citations. 168 passed, 6 skipped, 1 xfailed (9 new `env_to_yaml` tests). Tester spend about $0.17. Notes:
  - **P3-N1 (Medium, security):** keys are plain env vars on the service and in audit logs (shared A1 project). Fix option: Secret Manager (`--set-secrets`) plus key rotation. Needs the student's decision.
  - **P3-N2 (Major, Phase 4/7):** a warm broad question used 43.6 s of the 45 s worker timeout (long worker draft that synthesis rewrites); broad questions risk partial answers.
  - **P3-N3 (Minor):** an abstention ("evidence does not answer this") still cites unrelated chunks.
  - **P3-N4 (Minor):** cold start 17.9–34.4 s; index open 4.6–7.4 s on Cloud Run vs 0.4 s locally; image carries ragas/datasets/scipy the app never imports.
  - **P3-N5 (Polish):** no server log line for worker errors, blank log severity, `/favicon.ico` 404, public `/docs`, raw traceback on a missing key. The stdout/stderr comment in `main.py` was corrected.
  - **P3-N6/N7 (test hygiene, evidence caveat):** opening the index changes `chroma.sqlite3` bytes; the planner named target papers from its own knowledge, so rewording robustness of pure vector retrieval is not yet shown.
  - Not covered: fresh-project deploy (README lacks Cloud Build service-account IAM notes), concurrency. Phase 3 marked done.

### Phase 4: Routing (branch `phase-4-routing`, from `origin/main` at fb4fbe0)

- **Student decision recorded:** P3-N1 (keys as plain Cloud Run env vars, visible in `services describe` and deploy audit logs) is **accepted**. No Secret Manager. Recorded in README and compliance. Live test budget for Phase 4: $1.50 across both providers.
- **Models and prices re-verified 2026-10-27:**
  - `gpt-6.1-sol` returned by OpenAI `/v1/models`. The [OpenAI pricing page](https://developers.openai.com/api/docs/pricing) (standard tier) lists $2.00 in / $0.10 cached / $10.00 out per 1M tokens, and $4 / $15 over 272k context. That matches the OpenRouter listing.
  - `google/gemini-3.8-flash` in the OpenRouter catalog (`/api/v1/models`): $0.825 / $4.125 per 1M tokens, reasoning billed as output, 1,048,576 context, `reasoning` and `reasoning_effort` supported.
  - Cost ordering holds: Gemini is cheaper.
- **Spike findings (Gemini via OpenRouter, 150-word answer):**
  - Default settings: 507 of 703 output tokens were reasoning, TTFT 6.1 s.
  - `reasoning.effort=low` and `minimal`: 0 reasoning tokens, TTFT 2.5 s and 1.9 s.
  - `reasoning.enabled=false` is rejected with 400 "Reasoning is mandatory for this endpoint".
  - Streaming arrives in chunks of about 25 tokens with gaps of 0.3–0.5 s, so it is incremental once reasoning is controlled. The Phase 0 "3 chunks" result came from reasoning plus a one-sentence answer.
  - Chose `low` for all Gemini calls.
  - Tool calling and structured output work at low effort.
- **Upstream provider:** OpenRouter returns `provider` ("Google AI Studio") and `usage.cost_details`, but LangChain drops both. Added `OpenRouterChat`, a `ChatOpenAI` subclass that copies them into `response_metadata`. It attaches them only on the usage chunk, because LangChain concatenates string metadata across stream chunks (we saw `model_name` doubled). OpenRouter also reports `is_byok: true`: the account calls Google with the student's own key, so OpenRouter's `cost` is 0 and the upstream cost is reported separately.
- **Built:**
  - `app/llm/policy.py`: pure hybrid policy, dated price table, comparison cue, cost estimate.
  - `app/llm/router.py` rewrite: `decide`, `fallback_decision`, `call_llm` and `stream_llm` with per-attempt timeouts. SDK retries are off; one jittered retry on transient errors, then fallback to the other provider. Every attempt emits `llm_result` with status `ok`, `error` or `cancelled` (fixes P2-N4).
  - The worker loop, planner and synthesizer now call the router.
  - `done` gains `providers_used`, `fallbacks` and `est_cost_usd`.
  - Routing tab: per-turn header, decision inputs, reasoning effort, upstream provider, cost, fallback rows. Each answer gets a routing summary line.
  - `scripts/measure_routing.py`: before/after measurements.
  - Deploy preflight now requires `OPENROUTER_API_KEY`. The allowlist gains the routing settings; `ROUTER_FORCE_PROVIDER` is excluded, so the deployed service always routes by policy.
- **Latency (P2-N1/P3-N2):** levers were routing the planner and tool-loop calls to Gemini at low effort, plus a compact worker hand-off (≤ 8 bullets, ~200 words, ≤ 3 searches) instead of a prose draft. Measured locally, same three questions, 2026-10-27:

  | Question | Before (all OpenAI): worker span / TTFT / total | After (routed): worker span / TTFT / total | Synthesis provider after |
  |---|---|---|---|
  | Why do multi-agent LLM systems fail? | 45.0 s timeout ×2, **failed**, 96.6 s total | 11.3 s / 15.2 s / 20.5 s, done, 8 citations | Gemini (1 paper: MAST) |
  | How does Self-RAG decide when to retrieve? | 14.6 s / 20.0 s / 25.7 s | 11.7 s / 15.9 s / 19.2 s | Gemini (1 paper) |
  | Compare how Reflexion and Self-Refine use feedback… | 33.2 s / 39.9 s / 47.3 s (the draft call alone 24.3 s, 1,053 output tokens) | 8.5 s / 12.7 s / 20.0 s | OpenAI (comparison, 2 papers) |

  Gemini tool-loop calls take 1.3–2.1 s each, against 2.1–5.0 s on OpenAI. Synthesis TTFT is 1.3–1.7 s on both. The longest gap between Gemini synthesis chunks was 354 ms. One sample per question, local machine; Phase 9 measures properly. Raw events: `eval/results/phase4/{before,after}/`.
- **The baseline also reproduced P2-N4:** the two timed-out worker attempts left 2 `llm_route` events without results. After the fix, a unit test covers a worker timeout cancelling an in-flight call, and its route still gets `llm_result status=cancelled`.
- **Tests:**
  - `tests/unit/test_phase4_routing.py`: 38 tests covering the policy table and boundaries, overrides, the comparison cue, cost, retry → fallback, auth → direct fallback, both providers failing, fallback disabled, non-provider errors, per-attempt timeouts, cancellation, and the stream-fallback rules. Graph-level tests check that a single-paper question goes to Gemini and a multi-paper question to OpenAI.
  - Offline suite: 206 passed, 6 skipped, 1 xfailed.
  - `tests/integration/test_phase4_live_routing.py` (RUN_LIVE=1): 2 passed. Two ordinary questions picked different synthesis providers with all calls `ok`. An invalid Gemini model ID produced a real OpenRouter error, then a fallback to OpenAI.
  - I removed `DEFAULT_PROVIDER` from `Settings`, and one line in `tests/phase2_fakes.py` that passed it. That is an interface change, not a weakened test.
- **Bug in my own test:** a scripted `corpus_search` query of "q" failed the tool's 2-character minimum, so the worker failed and synthesis never ran. The test data was wrong, not the product.
- **Live spend so far:** about $0.29 (before/after runs $0.21, live tests about $0.06, spikes about $0.01).
- **Local browser check (headless Chrome, tester's `phase3_browser_deployed.py`):** 0 console errors. The routing line under each answer and the per-turn Routing tab rendered correctly: decision inputs, reasoning effort, "served by Google AI Studio", and cost.
- **Deploy (student approved), 2026-10-27 08:46 UTC:** revision `paper-analyst-00002-sbd`, image `paper-analyst:20261027-084606-fb4fbe0-dirty`, same settings as Phase 3. Smoke test passed (Gemini only, Self-RAG question, 0 failed attempts).
- **Deployed routing check (warm, `scripts/measure_routing.py`, `eval/results/phase4/deployed/`):**

  | Question | Synthesis | Worker span | TTFT / total | Citations | Est. cost |
  |---|---|---|---|---|---|
  | Compare Reflexion and Self-Refine feedback | OpenAI (capability: comparison, 2 papers) | 10.5 s | 14.6 s / 22.9 s | 8 | $0.024 |
  | Why do multi-agent LLM systems fail? (P3-N2 question) | Gemini (cost: 1 paper) | **8.4 s** (was 43.6 s in the Phase 3 tester run) | 13.0 s / 17.8 s | 5 | $0.013 |
  | Tree search vs. chain-of-thought trade-offs (unrehearsed) | OpenAI (capability) | 10.9 s | 14.9 s / 27.6 s | 10 | $0.037 |

  Every attempt returned `ok` with no fallbacks. Every Gemini call recorded the upstream "Google AI Studio". **Exit check met on the deployed URL:** two ordinary questions picked different synthesis providers, each with a visible rationale.
- **Live spend after deploy checks:** about $0.44 of $1.50.
- **Tester Phase 4 gate: PHASE PASS** (`docs/testing/phase-4-20261027T0855Z.md`). The exit check passed on the deployed URL with 8 unrehearsed questions: Gemini/cost for single-source questions and the greeting, OpenAI/capability for ReAct vs Toolformer and a 4-paper memory question. A headless browser on the deployed UI showed Routing-tab rows and routing lines that match the SSE stream. 0 route/result orphans; a real cancellation produced a `cancelled` result. Real fallbacks: an invalid OpenRouter key gave a 401, then OpenAI; a real 50 ms timeout gave a retry, then OpenAI. Deployed Gemini streaming is incremental. P3-N2 retest: worker spans 11.8–12.5 s against the 45 s timeout (resolved). Offline: 229 passed, 11 skipped, 1 xfailed. Tester spend about $0.55. Findings:
  - **P4-D1 (Major, open): Gemini cost estimates are about 45% too low.** OpenRouter's reported cost equals **$1.50 in / $7.50 out per 1M** on every BYOK call (I confirmed on my deployed events, e.g. 360 in / 112 out: estimated $0.000759, reported $0.00138). The catalog's $0.825 / $4.125 is not what Google bills on the student's own key. Routing is unaffected (still cheaper than gpt-6.1-sol). Affected: Routing-tab costs, `done.est_cost_usd`, README/architecture/.env.example prices, budget tracking. **Corrected Phase 4 spend estimate: about $1.05–1.10 of $1.50.**
  - P4-N1 (Phase 7): per-attempt timeouts are not capped by the remaining request budget (a hung synthesis can overrun 120 s by about 30–55 s).
  - P4-N2: only synthesis switches provider by features; planner and worker calls are always Gemini. Revisit when Summary and Data arrive.
  - P4-N3: the comparison regex misses wordings like "stack up against".
  - P4-N4 (tests): live tests share a pooled client across pytest event loops.
  - P4-N5: OpenAI synthesis took up to 23 s against its 60 s stream timeout.
- **P4-D1 fix (student asked for fix, retest and redeploy, 2026-10-27):**
  - Gemini price in `policy.PRICES` changed to $1.50 / $7.50 (cached $0.15).
  - Each `llm_result` now has `cost_usd` and `cost_source`. The reported cost is preferred; the dated estimate is used when the provider reports none (OpenAI). `done` reports `cost_usd` and `cost_basis`. The UI labels each cost "reported" or "est.".
  - **Pricing ambiguity found while fixing:** [Google's pricing page](https://ai.google.dev/gemini-api/docs/pricing) (checked 2026-10-27) lists Gemini 3.8 Flash at **$0.75 / $3.75 through 2026-12-31** and $1.50 / $7.50 from 2027-01-01, but OpenRouter's reported BYOK upstream cost uses $1.50 / $7.50. Without the student's Google billing data I cannot tell which is actually charged today. I chose the conservative, higher reported rate and documented the discrepancy (README, architecture §6, `.env.example`). The student can check the Google Cloud / AI Studio bill to settle it.
  - Tests: 3 new regression tests (price matches the observed reported costs; reported cost preferred and labelled; estimate used when nothing is reported). I also corrected one of my own earlier tests that hard-coded the old cached price. Offline suite: 232 passed, 11 skipped, 1 xfailed.
  - **Local verification:** the first rerun hit a local network outage. Every call failed on both providers (connection errors, OpenRouter timeout), and the app retried, fell back and ended `failed` honestly, with no faked answer (`eval/results/phase4/d1_fix_local_network_outage/`). Network checked with curl, then rerun: "What is the difference between ReAct and Reflexion?" ran Gemini planner and tool loop, then OpenAI synthesis (capability: comparison). Every Gemini `est_cost_usd` equals `reported_cost_usd` (for example 0.006387 = 0.006387); OpenAI is labelled estimated. Total $0.0336.
- **Redeploy after P4-D1 fix:** 2026-10-27 16:27 UTC, revision `paper-analyst-00003-vxz`, image `paper-analyst:20261027-162642-fb4fbe0-dirty`. Smoke test passed (Gemini, 0 failed attempts).
- **Tester retest after P4-D1: PHASE PASS, P4-D1 fixed** (`docs/testing/phase-4-retest-20261027T1645Z.md`).
  - Raw OpenRouter responses show `cost: 0`, `is_byok: true`, and an upstream cost at exactly $1.50 / $7.50.
  - Deployed revision 00003-vxz, unrehearsed question comparing Voyager and Generative Agents: Gemini reported $0.018264 and OpenAI estimated $0.012344, each equal to the token formula. `done.cost_basis` = {reported: 4, estimated: 1}. The UI labels each cost "(reported)" or "(est.)".
  - Offline: 236 passed, 11 skipped, 2 xfailed. Tester spend about $0.031.
  - The tester points out that OpenRouter's "reported" BYOK cost is its own tokens × rate calculation, not Google's invoice. Reported and estimated therefore agree by construction; the figure is a conservative upper bound.
  - New Low notes:
    - **P4-N6:** `_extra` should add `usage.cost` and `upstream_inference_cost` when `is_byok` (strict xfail TC-P4-14).
    - **P4-N7 (docs, before Phase 9):** mention that Google's Gemini free tier could apply; reword the `.env.example` "billed via BYOK" comment; in Phase 9, label Gemini cost as a dated upper bound and reconcile it with the Google bill.
- **Phase 4 live spend, final:** about $1.19 of $1.50 (corrected for P4-D1).
