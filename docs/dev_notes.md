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

### Phase 5: Tavily + Summary (branch `phase-5-tavily-summary`, from `origin/main` at 4005361)

- **Student brief (2026-10-27):** Tavily Remote MCP over Streamable HTTP with the key only as `tavilyApiKey`; a Summary agent with its own tools; the same `tavily_search` used by Research and Summary in different contexts; `W#` web citations; handle P2-N8, P3-N3 and P4-N2, plus the P4-N3/N6/N7 leftovers if cheap. Live test budget $1.50 (shared with the tester).
- **Spike:** re-listed the server's tools through `langchain-mcp-adapters` (`tavily_search`, `tavily_extract`, `tavily_crawl`, `tavily_map`, `tavily_research`). The current `tavily_search` args are `query`, `max_results`, `search_depth`, `topic`, `time_range`, `include_*` and others. The adapter returns MCP text blocks, plus `structured_content` in the artifact. Results have `url`, `title`, `content`, `score`; there is no reliable publish date, so we record retrieval time.
- **Low-support threshold:** calibrated `RAG_MIN_SCORE=0.55` on 16 queries. In-domain top-1 scores were 0.62–0.81, off-topic 0.18–0.26, and adjacent but uncovered topics (MCP authorization spec, "OpenAI this week") 0.50–0.52.
- **Built:**
  - `app/mcp_tavily.py`: `TavilyMCP` (process-wide, tool list cached, a session per call, 15 s timeout, one retry on connection errors), `parse_search_result` (success only with URLs; error payloads classified), and `shared_search_tool(ctx, agent, purpose)`, which registers `W#` and emits `web_search`.
  - `app/agents/summary.py`: `get_paper_outline`, `get_paper_section` (reading order, ≤ 3,500 tokens), `search_within_paper` (vector search filtered by `doc_id`), a paper resolver over read-only SQLite, and a status cue that enables the web.
  - Research: the shared tool behind a per-round gate (`needs_live` or best score < 0.55), traced as `tool_gate`.
  - Planner: `needs_live`, Summary in the worker list, and the out-of-domain rule (P2-N8). The fallback task is `needs_live`.
  - Redaction: `redact()`, `safe_error` unwraps ExceptionGroups, `RedactingFilter` on all handlers, and the `mcp` and `langchain_mcp_adapters` loggers at WARNING.
  - UI: `W#` viewer (snippet, retrieval time, finding agent, link out), "agent ⇄ MCP" trace rows, `tool_gate` rows, and dropped markers shown unlinked.
  - Deploy: `TAVILY_API_KEY` is now required by `deploy.sh`; the new settings are in the env allowlist.
- **P3-N3 root cause found:** when no worker cited anything, synthesis fell back to *every* chunk in the registry (`list(ctx.registry.by_cite.values())`), which is how unrelated chunks got attached to abstentions. Fixed: synthesis gets only cited evidence. Backstop: markers attached only to "evidence does not answer" clauses are not linked (`done.dropped_citations`).
- **P4-N2:** worker calls now switch to OpenAI (capability) when ≥ 3,000 tokens of full-section text (tools tagged `reads_full_text`) or ≥ 10,000 tokens of context are in context. P4-N3: the comparison regex gained "stack up", "outperform", "head-to-head", "as opposed to", "unlike" and "distinguish between". The tester's TC-P4-04 case that pinned the old miss now expects `True`. P4-N6: BYOK cost = `cost + upstream_inference_cost`; TC-P4-14's strict xfail was removed and it passes. P4-N7: free-tier wording in README, architecture §6 and `.env.example`.
- **Found while testing:**
  - The title resolver let "Agent" silently pick "AgentBench" (unique prefix). A fragment now wins only if it is the whole title or the name before the colon.
  - Tavily errors were re-wrapped three times ("TavilyUnavailable: TavilyUnavailable: …"); fixed.
  - In the first live sharing run, Research also searched Self-RAG's venue, which was Summary's task. Both roles now say "work only on your supervisor task".
- **Tests:** `tests/unit/test_phase5_tavily_summary.py` has 45 tests: parsing, bounds, key placement, redaction (including a real MCP client against a closed port, logs at DEBUG and SSE frames), the Research gate, partial-on-web-failure, sharing (same MCP object, two agents, distinct purposes, one `W#` namespace), Summary tools over the real local index, worker-side routing, P3-N3 and the planner fallback. Offline suite: 282 passed, 14 skipped, 1 xfailed. Live (`tests/integration/test_phase5_live_mcp_sharing.py`, RUN_LIVE=1): 3 passed.
- **Local real runs (2026-10-27, `eval/results/phase5/`):**
  - Live sharing test: Research ("find current developments on the topic") and Summary ("check the current status of a specific paper") both called `tavily_search` on `tavily-remote-mcp`. Synthesis used OpenAI (capability); total 52.9 s; $0.053.
  - Self-RAG retrieval question → Summary, no web; the final write-up switched to OpenAI at 3,289 section tokens. TTFT 31.6 s, total 34.2 s.
  - "Capital of Australia" → Research `needs_live` → 4 `W#` (Canberra), 14.7 s.
  - MCP authorization spec → corpus (1 paper) plus the official spec pages, 24.8 s.
  - Reflexion method summary → OpenAI from round 4 (3,317 section tokens), 32.1 s.
  - Fake Tavily key, real providers: 401 → `auth`, Research partial / `live_search_unavailable`, and the answer says web results could not be retrieved. With `--log-level debug`, the key and `tavilyApiKey` appear 0 times in the server log and the events.
  - Headless Chrome: `W1` opens the viewer (Princeton page, "retrieved … found by summary via tavily_search"), the Trace tab shows the MCP rows, 0 console errors, 9 distinct streamed lengths.
- **Latency observation:** Summary tasks that read 3 sections take 5 LLM rounds (TTFT about 25–32 s locally), slower than Research's 12–15 s and well within the 45 s worker timeout. Watch this in Phase 7/9.
- **Tavily pricing (checked 2026-10-27, docs.tavily.com/documentation/api-credits):** 1,000 free credits per month, a basic search costs 1 credit, pay-as-you-go $0.008 per credit. The plan of the student's key was not verified.
- **Live spend so far:** about $0.29 of $1.50 in LLM calls (recorded runs $0.24 + browser run about $0.04), plus about 15 Tavily credits.
- **Deploy (student approved), 2026-10-27 17:14 UTC:** revision `paper-analyst-00004-xsq`. The **smoke test failed**: the Self-RAG question went to Summary, `get_paper_outline` succeeded, then the next Gemini call returned HTTP 200 with **no text, no tool calls and 0/0 tokens** (upstream Google AI Studio). Summary ended `partial` / `citation_or_empty`, and synthesis abstained. The deployed rerun reproduced it. The same question run locally right after worked (outline plus three sections). Cause not determined; the cue was the empty usage. Fix: the router treats an empty completion as a transient `empty_response` (`llm_result status=error`, one retry, then fallback to the other provider). Two unit tests were added. Offline suite: 284 passed, 14 skipped, 1 xfailed.
- **Redeploy, 2026-10-27 17:23 UTC:** revision `paper-analyst-00005-zc8`, image `paper-analyst:20261027-172257-4005361-dirty`. Smoke test passed (5 citations, 0 failed attempts; no empty response this time). Startup logs: "live search enabled", index open 5.4–6.1 s.
- **Deployed exit check (warm, `eval/results/phase5/deployed/`): met.** The two-part memory + Self-RAG venue question planned Research + Summary, both `needs_live`. `research` called `tavily_search` once and `summary` twice, with distinct purposes and queries; Research did not repeat Summary's venue search. OpenAI capability synthesis, 53.4 s, $0.074. Current-information question (agent SDK releases in the past month): Research used `time_range=month` and cited 6 sources dated October 2026; Gemini synthesis, 21.1 s, $0.023.
- **Log scan:** 388 Cloud Run log entries since the deploy. Application logs contain 0 occurrences of the Tavily key or `tavilyApiKey`. The key appears in the 4 Cloud Audit Log `ReplaceService` entries from the two deploys (env vars, P3-N1, accepted).
- **Live spend before the tester:** about $0.49 of $1.50.
- **Tester Phase 5 gate: PHASE FAIL** (`docs/testing/phase-5-20261027T1735Z.md`). The exit check itself was verified on the deployed URL with the tester's own questions: a MetaGPT + frameworks question in an SSE capture had `summary` call `tavily_search` once and `research` twice, and a deployed headless Chrome run showed MCP trace rows for both agents. Current-info, off-topic, Summary routing switch (3,067 section tokens), `paper_not_found`, 31/31 C# and 33/33 W# resolution, fake-key handling, and 0 key shapes in 1,764 Cloud Run entries all passed. Two defects:
  - **P5-D1 (Medium):** an off-topic deployed answer (Everest boiling point) linked off-topic chunks C1 (0.22) and C6 (0.17) in "…papers do not contain…; thus, all figures are provided from current web findings [C1], [C6]…". The ";" split hid the abstention from the backstop (1 in 3 runs).
  - **P5-D2 (Medium, my regression):** `RedactingFilter` set `record.args=None`, and uvicorn's AccessFormatter unpacks args. On rev 00005, every access line raised "--- Logging error --- TypeError" (48/48) and was lost.
  - Notes: P5-N1 empty_response recurred once in about 12 deployed requests (handled by fallback; Phase 7); P5-N2 `status_cue` too broad; P5-N3 backstop unlinked supported "cannot address" claims; P5-N4 viewer shows 280 of 1,200 snippet characters (Phase 8); P5-N5 a MetaGPT status search returned basketball pages. Tester spend about $0.52 plus about 16 Tavily credits; about $0.48 of the phase budget left.
- **Fixes (local, 2026-10-27):**
  - **P5-D1:** a worker-cited corpus chunk with similarity < `LOW_RELEVANCE_SCORE` 0.35 (off-topic 0.16–0.26, in-domain top-k ≥ 0.58) is not given to synthesis and never linked; it goes to `done.dropped_citations`. The backstop also treats a "thus / so / therefore / hence / instead …" clause after a ";" that follows an abstention as abstaining for `C#` markers; web markers there still count as support.
  - **P5-N3:** an abstention now needs the evidence, papers or corpus as its subject.
  - **P5-N2:** `status_cue` now covers only publication-status words; recency is left to the planner's `needs_live`.
  - **P5-D2:** the filter redacts msg and each string arg in place and keeps the args tuple; it collapses only if a secret hides in a non-string arg. A local uvicorn run shows access lines formatted and redacted (`/health?tavilyApiKey=REDACTED`) with 0 logging errors.
  - TC-P5-01/02/17 now pass, and the strict xfails TC-P5-04/05 XPASSed, so their markers were removed as the tester instructed. Offline suite: 320 passed, 14 skipped, 1 xfailed.
- **Redeploy after P5-D1/D2 fixes (student approved), 2026-10-27 18:03 UTC:** revision `paper-analyst-00006-227`, image `paper-analyst:20261027-180301-4005361-dirty`. Smoke test passed (5 citations, both providers, 0 failed attempts).
- **Tester retest: PHASE FAIL** (`docs/testing/phase-5-retest-20261027T1810Z.md`). P5-D1, P5-D2, P5-N2 and P5-N3 were confirmed fixed on rev 00006: 0 C# linked on 3 off-topic questions; 0 "Logging error" in 64 entries, with access lines present and a canary key redacted; the sharing exit check was met again in the deployed browser. **New P5-D3 (Medium, caused by my P5-D1 fix):** the 0.35 cut-off also removed chunks Summary found with `search_within_paper`. Scoped keyword searches score 0.20–0.45, so a Voyager rerun silently lost its Experimental Setup citation, and `dropped_citations` was empty. Reproduced by TC-P5-18. Tester spend about $0.22.
- **P5-D3 fix:** the registry now records how each chunk was found. `register_chunk(c, scope="corpus"|"paper")` keeps the strongest corpus-wide score, and Summary's tools register `scope="paper"`. Synthesis withholds a chunk only if it was found solely by corpus search and its best score is < 0.35. Withheld ids are reported in `done.excluded_evidence`. Two regression tests were added; TC-P5-18 passes. Offline suite: 323 passed, 14 skipped, 1 xfailed.
- **Redeploy after the P5-D3 fix:** 2026-10-27 18:19 UTC, revision `paper-analyst-00007-kfs`, image `paper-analyst:20261027-181900-4005361-dirty`. Smoke test passed.
- **Tester retest 2: PHASE PASS** (`docs/testing/phase-5-retest2-20261027T1830Z.md`).
  - P5-D3 fixed on rev 00007: the Voyager question linked all 4 Summary citations, including the two scoped hits (0.285, 0.258) the old rule would have dropped. The Everest question and a paraphrase linked 0 C# and only this request's W#. Rev 00007 logs: 0 "Logging error", 0 credential shapes.
  - Offline: 324 passed, 14 skipped, 1 xfailed. The tester added TC-P5-19 (a mixed request: corpus off-topic chunk withheld, low-scoring in-paper chunk linked) and mutation-checked that TC-P5-18/19 catch the old rule.
  - Gaps the tester noted: the withholding path was not triggered on the deployment (the planner set `needs_live`, so no corpus search ran), so it is covered offline only. The two-agent `tavily_search` check was last run on rev 00006 (it passed there twice, plus once on rev 00005); rerun it at the final audit.
  - Open non-blocking: P3-N1 (accepted), P5-N1 (intermittent Gemini empty responses, Phase 7), P5-N4 (viewer snippet length, Phase 8), P5-N5 (one status search returned unrelated pages).
- **Phase 5 marked done.** Live spend for the phase: about $1.42 of $1.50 (me about $0.58 including smoke tests, tester $0.52 + $0.22 + $0.09), plus about 50 Tavily credits.

### Phase 6: Data (+ Code) (branch `phase-6-data`, from `origin/main` at 06f1141)

- **Student brief (2026-10-27):** a Data agent with `get_schema` and read-only, bounded `run_sql` (`D#` citations); a real Data → Summary hand-off through `depends_on`, verified on the deployed URL; feature routing for Data; Code agent only if time and budget allow; rerun the deployed two-agent `tavily_search` check (Phase 5 tester gap). Live budget $1.50 (shared with the tester).
- **Authorizer probe (SQLite 3.53.1):** pragma table-valued functions and `sqlite_master` show up as `SQLITE_READ` on their own names, so a table allowlist blocks them; `PRAGMA` and `ATTACH` have their own action codes; CTEs only produce reads of the base tables. `WITH RECURSIVE` needs `SQLITE_RECURSIVE` (allowed; bounded by the op cap and the wrapper LIMIT).
- **Built:**
  - `app/sqlguard.py`: lexical check, `mode=ro` + `query_only`, authorizer (SELECT, READ on 5 tables, allowlisted FUNCTION, RECURSIVE), `SELECT * FROM (\n<q>\n) LIMIT 51`, progress-handler cap (5M ops / 2 s), cell truncation, `schema_text`.
  - `app/agents/data.py`: Data agent, `DATA_ROLE`, `get_schema`/`run_sql`, one repair round (`MAX_SQL_FAILURES=2`), `sql_result` events, `paper_ids` for hand-offs.
  - `app/rag.py`: `register_sql` (`sql_row` per paper linked to the arXiv abstract page; `sql_result` for aggregate or empty results; each keeps SQL, columns and rows); `D` items in `format_evidence`; rule 4 of the augmentation prompt names `[D#]`.
  - `app/graph.py`: `data` worker; `upstream_papers` + `handoff` event at dispatch; `no_upstream_papers` skip rule; `n_data` synthesis feature; D# citation events carry SQL and rows; D# never treated as abstention-only.
  - `app/llm/policy.py` + `worker.py`: `repair_round` feature (set by `sql_error`/`sql_rejected`/`sql_timeout` tool errors) → OpenAI capability.
  - Prompts: planner rules for Data and Data-first hand-off plans; Summary works on handed-off papers; synthesizer reads `[D#]` as exact records.
  - UI: "data ⇄ SQLite" and `handoff` trace rows; D# viewer (SQL, row(s), abstract link); D# in the sources list and banner.
- **Found while building:**
  - My first guard never set `truncated`: the wrapper's LIMIT 50 meant the 51st row never existed. It now wraps with LIMIT 51 and returns 50.
  - `WITH x AS (…) DELETE …` passed the "starts with SELECT/WITH" check and was then blocked only as a syntax error inside the wrapper (`sql_error`), which would have offered the model a "repair" of a write. Added a keyword check outside literals so writes are `sql_rejected`.
  - Unbalanced `)` could close the wrapper early (`SELECT 1) UNION ALL SELECT … FROM (…`); the authorizer and LIMIT still applied, but the lexer now rejects it.
  - With an empty Data result, the dependent Summary would have run with nothing to summarise. The supervisor now skips it as `no_upstream_papers` and the request stays `done`.
- **Tests:** `tests/unit/test_phase6_data.py`, 72 tests: writes/PRAGMA/ATTACH/transactions, WITH-prefixed writes, internal tables and pragma functions, multiple statements, comments, wrapper escape, function allowlist, `mode=ro` without the authorizer, 50-row cap and truncation flag, cell cap, op cap (< 3 s), infinite recursive CTE, unknown column, empty result, joins/CTEs/windows, schema text, D# registry (per-row, aggregate, empty, stable ids, escaping), hand-off ordering, D# abstention exemption, graph-level hand-off injection with fakes, empty-result skip, repair round routed to OpenAI, second failure closes `run_sql`, injection via the question, Data-only synthesis on Gemini, uncited query = partial, `route_next` skip rule, planner/worker list. A module fixture asserts `papers.db` is byte-identical after the module. Offline suite: 396 passed, 14 skipped, 1 xfailed.
- **Live (RUN_LIVE=1) `tests/integration/test_phase6_live_data.py`: 6 passed** (hand-off end to end; repair round with an injected real `no such column` error, repair routed to OpenAI and answered 14; planner on 4 questions). `papers.db` sha1 `669db610…` before and after all live runs.
- **Local real runs (2026-10-27, `eval/results/phase6/local/`):**
  - Planner probe: 8/8 as intended (`planner_probe.txt`), $0.016.
  - "Which 2025 cs.MA papers … Summarize the two most recent": one 2025 cs.MA paper (MAS-GPT, correct per the DB); handoff t1 → t2 with `2503.03686 [D1]`; Summary read its outline and 3 sections; the answer says there is no second paper. Done, 35.6 s, $0.054.
  - "multi-agent-systems papers from 2025 or 2026 … two most recent": 6 rows D1–D6 (matches the DB), Summary read the two newest (2606.24535, 2606.19494). OpenAI synthesis; 33.4 s, $0.051.
  - Repos and categories: 58 papers (matches), per-category counts as `sql_result` D#; Gemini only, 13.6 s.
  - Injection ("DROP TABLE papers; UPDATE …; how many papers?"): the model only ran `SELECT COUNT(*)` → 89 [D1].
  - 2019 papers: empty result, Summary skipped `no_upstream_papers`, "The corpus contains no papers published in 2019 [D1]", status done.
  - ICLR acceptance: Data searched `comment`/`journal_ref` and found ReAct, AgentBench, SWE-bench.
  - Headless Chrome (`tests/e2e/phase6_browser_data_citations.py`): D1 opens the viewer with the SQL, the row and the arXiv abstract link; trace shows the SQLite and handoff rows; 0 console errors; 27 distinct streamed lengths.
- **Code agent: not built (my decision, recorded in architecture §5).** Data was done and verified, but Code needs repo snapshots, four tools, path-safety tests and an R# viewer; I kept the remaining budget for the deployed exit check, the sharing rerun and the tester. Three workers meet the floor.
- **Live spend before deploy:** about $0.30 of $1.50 (local runs $0.22, live tests ~$0.07, planner probe $0.016).
- **Deploy (student approved), 2026-10-27 19:10 UTC:** revision `paper-analyst-00008-nwt`, image `paper-analyst:20261027-190343-06f1141-dirty`, same settings. Smoke test passed (Gemini only, 5 citations, 0 failed attempts). Startup log: index ready, live search enabled, 5.00 s.
- **Deployed exit check (warm, `eval/results/phase6/deployed/`): met.**
  - "Which 2025 cs.MA papers are in the corpus? Summarize the two most recent" → plan data → summary (depends_on t1). SQL over `paper_categories` → 1 row D1; `handoff` t1 → t2 with `2503.03686`; Summary read the outline and 3 sections of that paper; it says there is no second paper. OpenAI synthesis (2 worker results); done, 28.5 s, $0.053, 7 citations, 0 unresolved/dropped.
  - Paraphrase "Which papers from 2026 about multi-agent systems are indexed? Give me a short summary of the two newest." → data (2 queries, D1–D5) → handoff of 3 ids → Summary read the two newest (2606.24535, 2606.19494). Done, 30.8 s, $0.046.
  - Data only: "Which authors appear on more than one paper?" → 74 authors (matches the DB), Narasimhan 8 papers (matches); Gemini synthesis (cost); 30.8 s, $0.060.
- **Deployed two-agent `tavily_search` rerun (Phase 5 tester gap), rev 00008:** the memory + Self-RAG venue question → Research ×2 ("find current developments on the topic") and Summary ×1 ("check the current status of a specific paper") on `tavily-remote-mcp`, 6 W# cited; done, 45.7 s, $0.039.
- **Deployed headless Chrome:** D1–D6 linked; the aggregate D1 viewer shows the SQL and result with no abstract link; trace shows the "data ⇄ SQLite" rows; 0 console errors.
- **Logs (rev 00008, 31 entries so far):** 0 "Logging error", 0 tracebacks, 0 credential shapes.
- **Live spend before the tester:** about $0.55 of $1.50.
- **Tester Phase 6 gate: PHASE FAIL** (`docs/testing/phase-6-20261027T1920Z.md`). The exit check itself worked on rev 00008 with the tester's own phrasings (data → summary on "the earliest 2024 paper", data → research), 12/12 unrehearsed planner phrasings, empty result, injection, repair (real model) and the Phase 5 sharing regression all passed. Tester spend about $0.37. Defects:
  - **P6-D1 (Major):** the guard bounded rows, operations and time but not memory. `WITH RECURSIVE … s||s …` built a 268 MB string in 1.19 s (2.14 GB peak) in few VM ops; one `printf('%200000000s', …)` built 200 MB. Reachable from the public URL (the tester ran a 3 MB `printf` there).
  - **P6-D2 (Minor):** Data rows without an `arxiv_id` column made the supervisor skip Summary as `no_upstream_papers`, and the synthesizer was told no papers matched.
  - Notes: N1 README still listed Code as a stretch worker; N2 corpus start date (ReAct is 2022-10-06); N3 a correct Data abstention ends `partial`; N4 calls after a successful repair stay on OpenAI (documented); N5 one 47 s Gemini span (Phase 9); N6 design_proposal still calls paper-to-code the signature workflow.
- **Fixes (local, 2026-10-27):**
  - **P6-D1:** `connect_ro` sets `SQLITE_LIMIT_LENGTH` = 100,000 bytes (and `SQLITE_LIMIT_SQL_LENGTH`); "too big" errors are `sql_rejected`. Over the limit, `printf`/`format` return NULL silently instead of failing, so both were removed from the function allowlist. Measured with `/usr/bin/time -l`: the doubling CTE (n<25), `group_concat` over a cross join, and ORDER BY / DISTINCT / GROUP BY over a million 99 KB rows are each refused or stopped in ≤ 2 s with 24–33 MB peak RSS. Found while probing: a recursive CTE with `UNION` (not `UNION ALL`) reads its own name and is refused by the table allowlist; it fails closed and can be repaired, recorded in architecture §5.
  - **P6-D2:** Data results now carry `sql_rows`; `found_no_papers` requires successful queries that returned zero rows. My `route_next` test data gained the new fields plus a titles-only case.
  - N1 (README) and N2 (architecture date range: 2022-10 to 2026-07 by v1 date) fixed. N3–N6 left as notes.
  - Offline suite: 419 passed, 34 skipped, 1 xfailed (all 18 tester Phase 6 tests pass, including TC T6-SQL-MEM ×3 and T6-NOPAPERS).
- **Redeploy after P6-D1/D2 fixes (student approved), 2026-10-27 20:38 UTC:** revision `paper-analyst-00009-bng`, image `paper-analyst:20261027-203235-06f1141-dirty`. Smoke test passed (Gemini only, 4 citations, 0 failed attempts).
- **Tester retest: PHASE FAIL** (`docs/testing/phase-6-retest-20261027T2050Z.md`). P6-D2 fixed, exit check (case H) passed on rev 00009, original D1 cases refused. **P6-D1 reopened:** (a) wide rows: one 90 KB value repeated in 400 columns reached 2.0 GB RSS in 0.26 s (fetchmany loaded rows before cells were cut); (b) temp spill: a GROUP BY over generated rows wrote ~946 MB of SQLite temp files in 0.49 s while RSS stayed at 25 MB; my 24–33 MB "sort" measurements counted RSS only, not temp files (in-memory on Cloud Run). N7: my architecture claim that a `UNION` recursive CTE is refused was wrong. Tester spend $0.106.
- **Fix (local):** the query now runs in a child process (`python -I -S -c …`, ~15 ms start) with `RLIMIT_AS` 512 MB, `RLIMIT_FSIZE` 16 MB, `RLIMIT_CPU` 5 s, `RLIMIT_NOFILE` 64, SIGXFSZ ignored, a private `SQLITE_TMPDIR`, and a parent kill after 8 s; plus `SQLITE_LIMIT_COLUMN` 32 and a row-by-row 1 MB raw-byte budget. Temp-storage, column and memory failures are `sql_rejected` with my own message; a child that dies without reporting is `sql_rejected`. macOS probe: legit joins and `SELECT *` fine (~30 ms); 40-col and 30-col wide rows, doubling CTE, GROUP BY and DISTINCT spills all refused in ~30 ms, child peak ~31 MB. N7 fixed in architecture. Offline suite: 421 passed, 34 skipped, 1 xfailed (tester's retest tests included).
- **Not verified locally on Linux:** Docker Desktop would not start any container (`docker run … echo hi` stayed in "Created"), so `RLIMIT_AS`/`RLIMIT_FSIZE` enforcement inside the Cloud Run sandbox is checked on the deployment instead.
- **Redeploy with the child-process guard (student approved), 2026-10-27 22:06 UTC:** revision `paper-analyst-00010-vpv`, image `paper-analyst:20261027-220039-06f1141-dirty`. Smoke test passed.
- **Deployed Linux check of the limits (`eval/results/phase6/deployed/rev10/`):**
  - A bounded spill probe (GROUP BY over 2,000 × 50 KB generated rows, about 100 MB of temp data, returning only lengths) was refused as `sql_rejected` "needs too much temporary storage": `RLIMIT_FSIZE` is enforced in the Cloud Run sandbox.
  - A 40-column query was refused by the column cap.
  - The Data → Summary question still works: data (2 queries) → handoff → summary, done, 6 citations, 30.7 s.
  - Each `run_sql` now takes about 0.5 s on Cloud Run (child start plus query) against about 30 ms locally.
  - Refused probe questions end with the templated "every planned step failed" answer, which is correct for a refusal.
- **Tester retest 2: PHASE PASS** (`docs/testing/phase-6-retest2-20261027T2230Z.md`).
  - Fixed and confirmed: P6-D1a (wide rows), P6-D1b (temp spill: refused after 16.8 MB in 0.036 s locally) and N7.
  - Cloud Run enforces `RLIMIT_FSIZE`: a 30 MB spill was refused, the 10 MB control accepted. `RLIMIT_AS` is unconfirmed there; the heaviest accepted query peaked at 61 MB.
  - Exit check on rev 00010 with a new paraphrase ("newest cs.CL paper … summarize its main contribution"): data → handoff 2607.26637 [D1] → summary.
  - Other checks: 3 concurrent deployed requests stayed isolated; the child env carries no credentials (canary check); no leftover temp dirs or child processes after 24 concurrent calls; `run_sql` 618–970 ms on Cloud Run.
  - Full suite: 424 passed, 34 skipped, 2 xfailed. Tester spend $0.089.
  - Open, non-blocking (student to decide):
    - **P6-D1c (Minor):** the temp limit is per file, so several materialized subqueries reached about 200 MB of temp storage locally; a two-table, ~40 MB version ran on the URL. Architecture §5 now states the per-file limit and this bound.
    - **P6-D3 (Minor):** an unexpected child failure (e.g. a missing DB) is reported as a resource refusal, its stderr is discarded, and it costs the model's repair round. Strict xfail `test_T6_CHILD_MISSING_DB_is_not_a_resource_refusal`.
    - **N8:** `SELECT count(*) FROM r` over a recursive CTE is refused as "read of 'r'".
    - **N9:** the refusal answer says "Please try again in a moment".
- **Phase 6 marked done.** Live spend for the phase: about $1.22 of $1.50 (me about $0.65 including smoke tests and deployed probes; tester $0.37 + $0.106 + $0.089), plus about 6 Tavily searches.

## Phase 7: Failure handling (branch `phase-7-failures`, 2026-10-27)

- **Scope:** worker wrappers, retries, budgets, a guarded fault-injection flag, tests; plus open notes P2-N7/P4-N1, P5-N1, P6-D3, N9 and P6-D1c. No latency work (Phase 8b).
- **Fault injection (`app/faults.py`):** `kind:name:mode[@n]` rules for workers, tools and providers. Per request via `X-Fault-Inject`, honoured only with a matching `X-Fault-Token` (server `FAULT_INJECT_TOKEN`, ≥ 24 chars); otherwise 403. `FAULT_INJECT` (every request) is local only: not in the `env_to_yaml.py` allowlist, and `deploy.sh` refuses it. Every hit is a `fault_injected` event; resulting events carry `injected: true`; answers say "[injected test fault, not a real outage]".
- **Budgets (P2-N7 / P4-N1):** a worker window (budget minus a 30 s synthesis reserve) for workers and the planner; every LLM attempt timeout capped by the time left; no attempt with < 0.25 s left (`budget_exhausted`); deadline cuts not blamed on the provider. The tester's P4-N1 test (`test_tc_p4_11`) documented the overrun; I changed it to assert the fix (wall ≤ budget + 0.5 s). It ran 12.50 s against the 12.5 s budget after the change.
- **Retries:** `empty_response` added to the worker-level transient kinds; no retry when both providers already failed in the request, or when the window is too short; each "not retrying" decision is a transition.
- **Partial / failed answers:** skip reasons ("it needs t1 (research), which failed with an internal error"); a deterministic "Incomplete evidence:" note after the streamed answer; the all-failed template's advice now matches the cause (N9: transient / the question / service side / injected); if synthesis fails before streaming, the workers' cited findings are returned (`partial`). A mid-stream failure still ends `failed` with a non-recoverable error (tester test T2-G10 unchanged).
- **P5-N1:** a synthesis stream that ends with no text is now `empty_response` (retry, then fallback) instead of a blank answer. Under `provider:gemini-openrouter:empty` the planner went Gemini empty ×2 → fallback → OpenAI ok, and later calls routed to OpenAI (`availability`).
- **P6-D3:** connection errors are caught inside the child (`sql_unavailable`); a non-zero exit without a result that was not killed by a signal is `sql_internal`, logged with exit code and redacted stderr tail. Neither uses the repair round; the Data task ends `data_unavailable`. Strict xfail removed from `test_T6_CHILD_MISSING_DB_is_not_a_resource_refusal` (now passes). Data task kinds now keep `sql_rejected` / `sql_timeout` separate from `invalid_sql`.
- **P6-D1c (fixed, cheap):** the child sets `PRAGMA temp_store=MEMORY` and `hard_heap_limit` 32 MB (64 MB first; lowered to stay near the tester's ~32 MB target). Tester probes: 4/6/8-table cases refused in about 0.05 s, 0 MB of temp files, child RSS ≤ 76 MB; ordinary queries unchanged. The single-sort 30 MB spill is now accepted (it fits the per-query heap).
- **UI:** `fault_injected` rows, `[injected test fault]` labels, skip reasons and attempt numbers in the trace; `#fault=…&fault_token=…` in the URL hash for browser demos, with a red banner.
- **Tests:** `tests/unit/test_phase7_failures.py` (41, MOCKED providers); `tests/integration/test_phase7_live_fault.py` (RUN_LIVE=1, real providers, through FastAPI). Changed my own P2-N4 test (`test_graph_worker_timeout_leaves_no_route_without_result`): the in-flight call now ends as a recorded `timeout` error (capped attempt) instead of `cancelled`. Full suite: 466 passed, 36 skipped, 1 xfailed.
- **Live local check (2026-10-27):** the cs.MA count + Self-RAG question with `worker:research:exception,worker:summary:exception` → plan data + summary; Data done (5 cs.MA papers [D1]); Summary failed (`worker_error`, injected, not retried); answer gives the count, says the Self-RAG part is not answered, and ends "Incomplete evidence: Summary (t2) failed with an internal error [injected test fault, not a real outage]". `partial`, 11.1 s, $0.012. The first run's planner chose Summary rather than Research for the named paper, so the test now injects into either content worker.
- **Latency effect:** none on failure-free requests (the same path plus cheap checks; the note is streamed only when evidence is missing, after the answer). The worker window can shorten worker time only when a request is already past about 90 s.
- **Deployed (student approved), 2026-10-28 ~00:50 UTC:** revision `paper-analyst-00011-c4t`, image `paper-analyst:20261028-004430-7afb300-dirty`; smoke test passed. Deployed checks (`eval/results/phase7/deployed/`): guard 403 (missing/wrong/empty token); partial case (Summary injected exception → partial, D1–D6, note); retry case (data timeout@1 → retry → done); empty case (Gemini empty ×2 → retry → fallback → done); headless Chrome banner + FAULT trace rows, 0 console errors; rev 00011 logs clean. My spend about $0.15.
- **Tester Phase 7 gate: PHASE PASS** (`docs/testing/phase-7-20261028T0100Z.md`). Own specs on rev 00011: data timeout (retry then "not retrying", partial), tavily_search error (partial), provider:*:down (honest failure 2.1 s, $0), summary hang@1 (real 45 s timeout, retry ok, 66 s), browser data exception (partial, 0 console errors); ordinary request unaffected; 7 guard variants all 403. Confirmed my three edits to existing tests are legitimate. Tester tests `tests/unit/test_phase7_tester.py` (16 pass, 2 strict xfail); full suite 482 passed, 36 skipped, 3 xfailed. Tester spend about $0.33. Open, non-blocking:
  - **P7-D1 (Minor):** a corrupt `papers.db` ("file is not a database") is `sql_error` → repair round, `invalid_sql`, "rephrase" advice. Strict xfails T7-S2/T7-S3.
  - P7-N1 hang-mode faults lose the injected label after the real timeout fires; P7-N2 "Planning…" stays after a 403 (Phase 8); P7-N3 token stays in `location.hash`/history (`history.replaceState`); P7-N4 422 echoes the whole spec; P7-N5 `deploy.sh` grep misses `FAULT_INJECT = x` (allowlist still blocks); P7-N6 planner error event lacks `error_kind`; P7-N7 cryptic startup error when the Chroma dir is missing.
- **Phase 7 marked done.** Live spend about $0.48 of $1.00.


## Phase 8: UI polish (branch `phase-8-ui`, from `origin/main` at 7b532d4, 2026-10-28)

- **Scope:** trace badges, source viewer, error states, Unicode, single-turn banner; redeploy. Open notes handled: P2-N3, P2-N5, P5-N4, P7-N2, P7-N3, the duplicated injected label. Optional notes done because they were cheap: P7-D1 and P7-N6. No latency work (Phase 8b).
- **Frontend (architecture §10):**
  - **`md.js` (new):** Markdown → a plain data tree, then DOM via `createElement`/`textContent`. It handles headings, lists, `---`, quotes, fences, tables, bold, italic, code, and links for http(s) only. HTML in model output stays literal text. LaTeX still renders raw (out of scope).
  - **Trace:** grouped per turn and per agent with `<details>`. Status badges are done, partial, failed, skipped and injected. Each turn has a flow strip, every row has a `+t` time, and transitions are written as sentences. The injected label appears once per row.
  - **Viewer:** title from the paper title and section (P2-N3); the full web snippet in a scrolling panel (P5-N4); an abstract link. Dark-mode link colour `--link` gives 9.75:1 contrast.
  - **Errors (P7-N2):** HTTP errors, closed-before-done streams and a 60 s no-byte watchdog (`AbortController`) all settle the turn on a visible error. The fault token is cleared with `history.replaceState` (P7-N3).
  - **Per turn:** a status badge, a clock and routing chips; the status line says "Streaming answer… (first token after X s)". The single-turn banner now says "no conversation memory". `dir="auto"` and an IME-safe Enter for CJK input.
  - **Bug found on the deployment and fixed:** the new chat header class `.turn-head` clashed with the Routing tab's `tr.turn-head`, which squeezed the turn row into one column. Renamed to `.turn-bar`. The browser test now checks that the row spans the table.
- **Server (small):**
  - `citation` events carry `title`, `section`, and the full `snippet` for web results. `web_search` sends the full snippet (`TRACE_SNIPPET_CHARS` 280 removed).
  - The planner `error` has `error_kind="planner_error"` (P7-N6).
  - A corrupt `papers.db` ("not a database") is `sql_unavailable`, not `sql_error` (P7-D1). The tester's strict xfails T7-S2 and T7-S3 now pass, so their markers were removed as the tester instructed.
- **Not done (optional):** P7-N1 (hang faults lose the injected label after the real timeout). The `TimeoutError` comes from `wait_for` outside the fault, so the label would need changes across the router, planner, worker and synthesis paths. P7-N4 and P7-N5 not done either.
- **Tests:**
  - `tests/unit/test_phase8_ui.py` (8): md.js under Node via `tests/unit/js/md_check.mjs`, static safety, fault-token clearing, the HTTP-error path, citation title/section, and the full web snippet.
  - Full suite: 492 passed, 36 skipped, 1 xfailed (before: 482/36/3).
- **Browser:** `tests/e2e/phase8_browser_ui.py` (real headless Chrome, CDP). Local run 19/19. Deployed run on rev 00013: **20/20** (`eval/results/phase8/browser_report.json`, 11 screenshots), covering:
  - 403 error state, token cleared, banner;
  - 69 answer-length changes before `done`;
  - routing chips and Routing tab with both providers;
  - grouped trace with badges;
  - C1, D1 and W1 viewers;
  - a French/Chinese/emoji question and answer intact;
  - dark-mode contrast, no console errors.
- **Deployed (student approved), 2026-10-28:** rev `paper-analyst-00012-452` (image `…:20261028-013101-29c6c50`), then the CSS fix as rev `paper-analyst-00013-v4k` (image `…:20261028-013849-c25d730-dirty`; "dirty" only because `eval/results/phase8/` was untracked). Smoke tests passed both times. The second smoke test saw a Gemini `empty_response` that was retried and then succeeded (P5-N1 behaviour, handled).
- **Spend so far:** about $0.23 of the $0.75 phase budget (local browser run $0.07, two smoke tests about $0.02, two deployed browser runs about $0.14), plus about 8 Tavily credits.
- **Tester Phase 8 gate: PHASE PASS** (`docs/testing/phase-8-20261028T0150Z.md`), on rev 00013. Tester spend about $0.19 plus 2 Tavily searches, so the phase total is about $0.42 of $0.75. What it checked:
  - streaming in the browser (client and server token timings match);
  - routing rows and chips against the routing events;
  - all 16 citation links opening the right viewer;
  - two tabs at once kept separate;
  - a real mid-stream disconnect through a proxy;
  - a local injected Data fault;
  - a mock-backend suite of hostile Markdown/HTML, which executed nothing.
  - **P8-D1 (Minor):** pressing Enter during a running turn starts a second paid request (`requestSubmit` bypasses the disabled button).
  - **P8-D2 (Minor):** a long unbroken token in a question pushes the Routing-tab columns past the panel edge (`tr.turn-head td` has no `overflow-wrap`).
  - Notes:
    - N1: light-mode contrast is 4.49:1 on the 11 px green badges.
    - N2: LaTeX shows raw.
    - N3: the model the API returned is not shown.
    - N4: in a hidden tab, `requestAnimationFrame` pauses painting.
    - N5: a chunk id containing `/` would 404; no current id has one.
  - Retest without spend: `tests/e2e/phase8_tester_mock_ui.py`.
- **Fixes after the Phase 8 gate (local, 2026-10-28, no LLM spend):**
  - **P8-D1:** a `busy` flag makes the submit handler return while a turn is running.
  - **P8-D2:** table cells get `overflow-wrap: anywhere`.
  - **N1:** light `--ok` is now #236a40, and `.cite-dropped` uses a solid muted colour instead of opacity.
  - Result: the tester's `phase8_tester_mock_ui.py` passes 27/27, including M7b, M12 and M9. Full suite: 492 passed, 36 skipped, 1 xfailed.
  - Not yet redeployed (needs approval).
- **Tester retest: PHASE PASS** (`docs/testing/phase-8-retest-20261028T0216Z.md`) on rev 00014 (image `…-a02dd08`).
  - P8-D1, P8-D2 and N1 were confirmed fixed, including on the live page with real Enter key events and no LLM spend.
  - The deployed smoke run passed, with both providers used. Tester spend about $0.15.
  - New **P8-D3 (Minor):** the global `td { overflow-wrap: anywhere }` from the P8-D2 fix broke "Turn" and "#n" mid-word.
  - **Fix (local):** the Routing table uses `table-layout: fixed` with set column widths. Wrapping anywhere is limited to data cells and the turn row; the Turn column and its header stay whole.
  - Results: the tester's mock suite (now with M12b) passes 28/28 (`eval/results/testing/phase8-retest3-20261028T0227Z/mock/`); the full suite still passes.
- **Deployed (student approved), 2026-10-28:** rev `paper-analyst-00014-2rj` (image `…-a02dd08`) with P8-D1, P8-D2 and N1, then rev `paper-analyst-00015-zhx` (image `…-16f02eb`, clean tree) with P8-D3. Smoke tests passed both times.
- **Tester retest 2: PHASE PASS** (`docs/testing/phase-8-retest2-20261028T0245Z.md`) on rev 00015.
  - Results: mock suite 28/28; no-spend live checks 10/10; one real query with both providers, rows matching their events and incremental streaming; the Routing table stays inside the panel with "Turn" and `#n` unbroken in light and dark at 1500 px and 1100 px.
  - New non-blocking **P8-N6:** at a 1100 px window the Agent column shrinks to 73 px and "synthesizer" wraps mid-word.
- **Phase 8 marked done.** Live spend about $0.61 of $0.75 (me about $0.26, tester about $0.19 + $0.15 + $0.03), plus about 12 Tavily credits.

## Phase 8b (latency, package B) — in progress, 2026-10-28
- Student chose package B (options 1, 2, 3, 4, 6, 7, 10); option 8 (parallel dispatch) held back pending approval.
- Baseline rev 00015 (warm, 1 run each, `eval/results/phase8b/baseline-rev00015/`): simple TTFT median 18.1 s, multi 28.1 s.
- Built (branch `phase-8b-latency`, uncommitted): planner `papers`/`queries` hints → seeded first tool calls; speculative corpus search during planning (kept only for a single Research task); evidence-only single-task plans (no worker write-up; synthesis answers from gathered evidence, off-topic filter kept); synthesis capability rule for ≥3k tokens of raw sections; Gemini `minimal` effort except planner (`low`); trace UI rows for seeds/speculation/evidence-only. Switches: PLANNER_SEEDS, EVIDENCE_ONLY_SINGLE_TASK, SPECULATIVE_RETRIEVAL, GEMINI_PLANNER_REASONING_EFFORT. Tool-call bounds unchanged (3/round, 4/task).
- Tests: 435 unit passed (13 new in `tests/unit/test_phase8b_latency.py`). Bug found and fixed: questions > 300 chars are not searched speculatively.
- Local measurement (15 simple runs, `eval/results/phase8b/local-pkgB-{1,2}/`): simple TTFT median 7.6 s (4.8–16.4 s); outliers are the planner reasoning at `low` (10–12 s, 3/21 calls). Multi (Data → Summary) still 22–32 s. Both providers used in ordinary queries; cost per simple query ~$0.015–0.034 (was ~$0.04–0.05).
- Option 10 done: ragas and its 25 eval-only dependencies moved to `requirements-eval.in/.txt` (exact pins, resolved against the runtime lockfile; runtime versions unchanged). Local image 1.86 GB → 1.12 GB (arm64 build); container runs as uid 10001, honours PORT, ready in 2.9 s locally.
- Routing: a synthesis rule (≥ 3k raw section tokens → OpenAI) was tried and dropped. It broke the tester's Phase 4 live test (an ordinary single-paper question and a comparison must pick different providers); the 6k-evidence rule already covers long evidence.
- Speculative search: a discarded search is now awaited (≤ 1 s) instead of cancelled mid-request. Known test-harness issue: `test_phase6_live_data.py::test_planner_routes_metadata_to_data_live` fails intermittently with "Event loop is closed". langchain-openai shares one HTTP client per process, and each anyio test has its own loop; the planner now makes an embedding call. It passes alone and the server uses one loop; left for the tester (e.g. SPECULATIVE_RETRIEVAL off in planner-only tests).
- Live tests also overwrite committed evidence files in eval/results/phase5-7; restored with git checkout after each run.
- RAGAS faithfulness (`scripts/faithfulness_check.py`; evaluator gpt-6.1-sol; contexts = cited sources): baseline rev 00015 mean 0.959 (n=7) vs local package B (pkgB-1, with the since-dropped rule) mean 0.989 (n=7). One sample per question.
- Committed 7b0dfe8 on `phase-8b-latency`. Deploy blocked by the auto-mode permission check; the student needs to run `scripts/deploy.sh`. Then: warm measurement (≥ 5 simple runs), faithfulness on deployed runs, a real cold start, and the tester in phase mode.
- Deployed rev 00016 (student ran deploy.sh). Warm, 2 passes: simple TTFT median 6.35 s (n = 10, 4.6–8.3 s), Data → Summary 18.2/19.2 s; RAGAS faithfulness mean 0.985 (n = 14). Results in docs/latency_plan.md and eval/results/phase8b/deployed-rev00016-warm/. Cold start deferred to the next phase (student decision). Next: tester phase run.
- Tester phase run (docs/testing/phase-8b-20261028T0454Z.md): PHASE BLOCKED. The OpenAI account ran out of credits mid-run; the student added credits.
  - P8b-D1 (test harness) fixed in tests only. My unit test no longer makes a real embedding call (speculation off where the real retriever is used). New tests/integration/conftest.py fixture `close_openai_clients` (used by the six live modules) drops langchain-openai's cached clients and collects them at the end of each test while its loop runs. A module-scoped anyio_backend was tried first and did not help (anyio still makes a loop per test). Unit suite 5/5 clean (446 passed); live Phase 6 module 4/4 clean (was about 1 in 3 failing); Phase 4 live routing passes.
  - P8b-D2 fixed in app code (router/context/graph): real provider errors (no fault fired in that attempt, not budget_exhausted) are recorded per request. A task is labelled injected only if no real error happened during it; otherwise its line says it was a real provider error with a test fault also active, and the advice covers the real cause. Two new unit tests. Not deployed yet (next redeploy).
  - The student chose not to run the retest and accepted the block (2026-10-28). Phase 8b marked done as an accepted block. Open notes from the tester: N1 (insufficient_quota retried as a rate limit), N2 (Research does not unlock web search when corpus_search errors), N3 (HyDE seed note wording), N5 (6.35 s just above 6 s), N6 (faithfulness check is a regression signal only).

## Phase 9 (evaluation) — in progress, branch `phase-9-eval`, 2026-10-27
- Start state: live service still on rev `paper-analyst-00016-dnl` (no P8b-D2 fix). Redeploy requested from the student (auto mode cannot run `scripts/deploy.sh`).
- OpenAI credits checked with one embedding and one `gpt-6.1-sol` call: both succeeded.
- Test set draft `eval/testset/testset_v1.jsonl` (36 items, 26 RAGAS); construction method in `eval/testset/README.md`. The first sampler run produced three table- or caption-only chunks, so a prose filter and an appendix filter were added before any question was written. All 40 reference chunk ids resolve. Waiting for student review, then freeze.
- Planner effort probe (`scripts/planner_effort_probe.py`, `eval/results/phase9/planner-effort-probe/`): 19 questions × 2 runs × {low, minimal}.
  - Both 38/38 correct plans; worker plans identical on all 38 pairs.
  - Planner latency median 2.54 s (low) vs 2.51 s (minimal); p90 3.27 vs 3.34 s; max 4.4 vs 6.7 s; one reasoning call each.
  - No measurable gain from `minimal`; decision left to the student. Cost about $0.15.
- Student decisions (2026-10-27): planner Gemini effort stays `low`; option 8 (parallel dispatch) not added, so multi-agent questions are reported as they are, with option 8 noted as future work. Redeploy started by the student.
- Test set v1 frozen 2026-10-28 after review round 1 (44 items, 32 RAGAS), sha256 50e9c0f3e629e9bc39f6412e72133ab8c925b61f4eeaf22e791680cf1f8f5382.
- Pilot (local, forced OpenAI; `eval/results/phase9/pilot-forced-openai/`):
  - c01 scored faithfulness 1.00, answer relevancy 0.91, context precision 0.82, context recall 1.00 (19 contexts).
  - s01 failed with `partial`: Summary resolves papers by title only, and "MMOA-RAG" is not in its paper's title. Same family as tester note N3 (HyDE).
  - Judge cost about $0.15 for one item with 19 contexts; context precision makes one call per context.
  - The student raised the Phase 9 budget to about $9 (option A: full plan) and chose to fix the lookup before scoring.
- Fix (`app/agents/summary.py` `resolve_paper`): when no title matches, the paper whose *abstract* names the method as a whole word is used.
  - Only abstracts count, because body text also names baselines (R1-Searcher appears only as a TeaRAG baseline).
  - Several matching abstracts give an ambiguous error with candidates; names of 4 characters or fewer must match case exactly.
  - Covers the planner seeds too, so N3 (HyDE) is fixed as well. The outline says when a paper was resolved this way.
  - 10 new unit tests; full suite 529 passed, 36 skipped, 1 xfailed.
  - Local rerun of s01 (`pilot-forced-openai-fix/`): done, reference chunk at rank 2, correct cited answer. Needs a redeploy (rev 00018).
- Cold start, attempt 1 (rev 00017, `eval/results/phase9/cold-start-1/`):
  - Scale-to-zero confirmed in the logs: the last request was the deploy smoke test at 06:15:02; uvicorn shut down at 06:30:15; no requests in between.
  - The gate in my wait script wrongly counted the smoke-test request itself (timestamp-filter bug), so I re-checked the logs and sent the request by hand at 06:40.
  - `measure_latency.py` sends `/health` before the question, so `/health` took the cold start: 21.8 s.
    - "Starting new instance" at 06:40:02.4; uvicorn process at 06:40:18.8; app startup 5.76 s (index open); ready at 06:40:24.6.
    - The question then ran warm: TTFT 6.3 s, total 9.1 s.
  - This is a valid **container cold-start** sample (21.8 s to first response) but **not** a cold TTFT for a question. Added `--no-health`; the rev 00018 cold run sends the question first.
