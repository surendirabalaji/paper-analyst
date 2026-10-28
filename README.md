# Agentic AI Paper Analyst

A multi-agent system for CSCI 599 Assignment 2. It answers questions about a corpus of arXiv papers on LLM agents and RAG. A LangGraph supervisor delegates to specialized agents (Research, Summary and Data; a Code agent was planned as a stretch goal and not built). They use RAG over a Chroma index, live web search through Tavily Remote MCP, and generation routed between two providers (OpenAI and Gemini via OpenRouter). Answers stream to a vanilla HTML/JS page over SSE, with the routing decisions, an orchestration trace and clickable citations.

> **Status: under construction (phase-wise build).** Sections marked *TBD* are filled in as phases complete.

- **Live URL:** https://paper-analyst-g7b7dxmyuq-uw.a.run.app (Cloud Run, us-west1; first deployed 2026-10-27)
- **Architecture:** [docs/architecture.md](docs/architecture.md)
- **Evaluation:** [docs/evaluation.md](docs/evaluation.md)
- **Build plan:** [docs/plan.md](docs/plan.md)
- **State:** stateless and single-turn. Each question is answered independently; nothing is stored between requests.

## Providers and services

| Purpose | Service | Model / endpoint |
|---|---|---|
| Generation (provider 1) | OpenAI API (direct) | `gpt-6.1-sol` (verified 2026-10-27) |
| Generation (provider 2) | Google Gemini via OpenRouter | `google/gemini-3.8-flash` (verified 2026-10-27) |
| Embeddings | OpenAI API | `text-embedding-3-small` |
| Live web retrieval (shared MCP tool) | Tavily Remote MCP, Streamable HTTP | `https://mcp.tavily.com/mcp/?tavilyApiKey=…` (key only as the query parameter; tool `tavily_search`, shared by Research and Summary) |
| Vector store | Chroma (embedded, baked into the image) | |
| Hosting | Google Cloud Run | |

**Routing.** Each LLM call is routed in code by a hybrid cost/capability policy (`app/llm/policy.py`). Planning, tool-loop and simple single-source synthesis calls go to Gemini (cost). Synthesis that reconciles several papers or worker results, or answers a comparison question, goes to OpenAI (capability). On a provider error the call is retried once (transient errors only) and then falls back to the other provider, traced as a separate `fallback` event. Every decision appears in the UI's Routing tab and in a one-line summary under each answer. Full table: [docs/architecture.md §6](docs/architecture.md). Routing settings in `.env`: `GEMINI_REASONING_EFFORT` (default `low`), `OPENAI_REASONING_EFFORT` (empty = model default), `LLM_FALLBACK` (1), `LLM_TIMEOUT_S` (60). `ROUTER_FORCE_PROVIDER` is for evaluation runs only and is not deployed.

**Live retrieval and MCP tool sharing.** Research and Summary each get a handle on the same Tavily Remote MCP `tavily_search` tool. Research uses it for current developments on a topic. It is unlocked when the planner marks a task `needs_live`, or when the best corpus similarity is below `RAG_MIN_SCORE` (0.55), which covers questions outside the corpus. Summary uses it to check one named paper's current status (venue, acceptance, follow-ups). Web results are cited as `W#` with their URL and retrieval time. Every call appears in the Trace tab with the calling agent. If Tavily fails (bad key, quota, network, no results), the answer continues from the papers and says that live results were unavailable. Settings: `TAVILY_API_KEY`, `TAVILY_TIMEOUT_S` (15), `TAVILY_MAX_RESULTS` (4), `RAG_MIN_SCORE` (0.55). Details: [docs/architecture.md §8](docs/architecture.md).

## Local setup

```bash
uv python install 3.13
uv venv -p 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env   # then fill in the keys
```

Dependencies: `requirements.in` lists the top-level runtime packages; `requirements.txt` pins every package with `==` and is what the Cloud Run image installs. Evaluation-only packages (RAGAS and its dependencies) are in `requirements-eval.in` / `requirements-eval.txt`, also pinned with `==` and resolved against the runtime lockfile, so the image stays smaller (Phase 8b). Install both to run tests or the evaluation:

```bash
uv pip install -r requirements.txt -r requirements-eval.txt
```

Regenerate the lockfiles with:

```bash
uv pip compile requirements.in --python-version 3.13 \
  --python-platform x86_64-unknown-linux-gnu -o requirements.txt
uv pip compile requirements-eval.in -c requirements.txt --python-version 3.13 \
  --python-platform x86_64-unknown-linux-gnu -o requirements-eval.txt
```

## Corpus ingestion

The corpus is defined by `data/corpus_seed.json` (committed list of arXiv IDs, topic groups and selection method). `scripts/ingest.py` builds everything from it:

```bash
.venv/bin/python scripts/ingest.py --dry-run   # fetch + parse + chunk; prints counts and embedding cost, writes nothing
.venv/bin/python scripts/ingest.py             # build or incrementally update the index (needs OPENAI_API_KEY)
.venv/bin/python scripts/query_index.py "how does Self-RAG decide when to retrieve?"   # retrieval smoke test
```

- **Source:** arXiv LaTeXML HTML (`https://arxiv.org/html/<id>v<n>`) plus arXiv API metadata. Requests go through one connection with a 3.5 s minimum gap and back off on HTTP 429 (arXiv's API terms: at most one request every 3 s). A cold fetch of the full corpus takes several minutes; raw HTML is cached in `data/raw/` (git-ignored).
- **Outputs:** Chroma index in `data/chroma_index/` (collection `arxiv_agents`), SQLite metadata in `data/papers.db` (`papers`, `authors`, `paper_authors`, `paper_categories`, `repos`), and the provenance manifest `data/corpus_manifest.json` (committed). The index and DB are git-ignored and excluded from the submission zip; they are baked into the container image.
- **Re-running is safe:** chunk IDs are deterministic (`{doc_id}::{section_id}::{nnn}`) and unchanged chunks are skipped by content hash, so a second run embeds nothing and adds 0 chunks. Paper versions are pinned by the manifest; `--update-versions` moves to newer arXiv versions (old-version chunks are deleted). `--rebuild` drops the collection and re-embeds everything (required if the embedding model changes; the script refuses to mix models).

## Run locally

Needs the built index (`data/chroma_index`, see above) and `.env` with `OPENAI_API_KEY` and `OPENROUTER_API_KEY` (the router uses both providers).

```bash
.venv/bin/uvicorn app.main:app --port 8080
# open http://localhost:8080 ; health check: curl localhost:8080/health
# raw SSE stream:
curl -N -X POST localhost:8080/api/chat -H 'content-type: application/json' \
     -d '{"query": "How does Self-RAG decide when to retrieve?"}'
```

Endpoints: `POST /api/chat` (SSE stream of `meta`, `plan`, `transition`, `llm_route`, `llm_result`, `tool_call`, `tool_result`, `retrieval`, `worker_result`, `token`, `citation`, `error`, `done`; each event carries `request_id`, `seq`, `ts`), `GET /api/source/chunk/{chunk_id}`, `GET /health`, `GET /`.

The app is single-turn: each question is answered independently.

## Run in a container (local)

```bash
docker build -t paper-analyst .
docker run --rm -p 8080:8080 --env-file .env paper-analyst      # PORT defaults to 8080
docker run --rm -e PORT=9090 -p 9090:9090 --env-file .env paper-analyst   # Cloud Run-style injected PORT
```

The image (multi-stage `python:3.13-slim`) installs the pinned requirements into `/opt/venv`, runs as the non-root user `app` (uid 10001), and starts `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}`. It bakes in `data/chroma_index/` and `data/papers.db`, so build the index first (see *Corpus ingestion*). The index directory is owned by uid 10001 because Chroma opens `chroma.sqlite3` read-write even for queries; everything else is root-owned and read-only. Secrets are never copied into the image: `.dockerignore` and `.gcloudignore` are allowlists, and keys are passed at run time.

## Deploy to a fresh GCP project

Prerequisites: the [gcloud CLI](https://cloud.google.com/sdk/docs/install), a GCP project with billing enabled (or credits), the built index in `data/` (see *Corpus ingestion*), and a filled-in `.env` (`OPENAI_API_KEY`, `OPENROUTER_API_KEY` and `TAVILY_API_KEY` are required; `scripts/deploy.sh` refuses to run without them).

```bash
# 1. Authenticate and pick the project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Enable the three APIs the deploy needs (one time per project)
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

# 3. Build with Cloud Build, deploy to Cloud Run, smoke-test the URL
GCP_PROJECT_ID=YOUR_PROJECT_ID scripts/deploy.sh            # asks for confirmation; --yes skips it
GCP_PROJECT_ID=YOUR_PROJECT_ID scripts/deploy.sh --dry-run  # print the commands, change nothing
```

`scripts/deploy.sh`:

1. Checks the tools, the index files, the active gcloud account and that the three APIs are enabled (it does not enable them for you).
2. Converts an allowlist of `.env` keys into a Cloud Run `--env-vars-file` (`scripts/env_to_yaml.py`), written with mode 0600 into a `mktemp` directory **outside the repo** and deleted on exit. Placeholder values (`REPLACE_ME`) are rejected.
3. Creates the Artifact Registry Docker repo `paper-analyst` in the region if it does not exist.
4. Runs `gcloud builds submit` (linux/amd64 build, so it also works from Apple Silicon). `.gcloudignore` uploads only 31 files: the Dockerfile, requirements, `app/`, `ingest/`, the Chroma index and `papers.db`.
5. Deploys the service with `--allow-unauthenticated --memory 2Gi --cpu 1 --cpu-boost --concurrency 10 --timeout 300 --min-instances 0 --max-instances 3`.
6. Smoke-tests `GET /health` and one streamed `/api/chat` corpus question (requires tokens, citations and a `done` event with no `error`).

Overrides (environment variables): `GCP_REGION` (default `us-west1`), `CLOUD_RUN_SERVICE` (`paper-analyst`), `AR_REPO` (`paper-analyst`), `ENV_FILE` (`.env`), `MEMORY`, `CPU`, `CONCURRENCY`, `TIMEOUT`, `MIN_INSTANCES`, `MAX_INSTANCES`.

Credentials end up as plain Cloud Run environment variables on the service, not in the image or the repo. Project members with Cloud Run access can read them, and they also appear in the Cloud Audit Log entries for each deploy. Secret Manager was considered (tester note P3-N1); the student accepted plain env vars for this project on 2026-10-27. To rotate a key, edit `.env` and rerun the script.

Time a deployed instance from the client (health, TTFT, total): `.venv/bin/python scripts/probe_remote.py https://YOUR-SERVICE-URL`.

## Evaluation

*TBD (Phase 9)*

## Cost and access (dated)

Generation prices, checked **2026-10-27** (per 1M tokens, input / output). Re-check before submission:

| Model | Price | Source |
|---|---|---|
| `gpt-6.1-sol` (OpenAI direct) | $2.00 / $10.00 (cached input $0.10; long context over 272k tokens $4.00 / $15.00) | [OpenAI API pricing](https://developers.openai.com/api/docs/pricing), standard tier |
| `google/gemini-3.8-flash` (via OpenRouter, BYOK) | **$1.50 / $7.50** used for costing (reasoning tokens billed as output) | OpenRouter's reported per-call upstream cost matches this exactly. [Google's pricing page](https://ai.google.dev/gemini-api/docs/pricing) lists it as the standard rate from 2027-01-01 and $0.75 / $3.75 through 2026-12-31. The [OpenRouter catalog](https://openrouter.ai/api/v1/models) lists $0.825 / $4.125, which does not apply to BYOK calls. We use the conservative reported rate until a Google bill confirms which applies. The calls run on the student's own Google key, so that key's Gemini free tier may cover some calls; treat Gemini figures as an upper bound. |
| Tavily search (basic depth) | 1 API credit per search; 1,000 free credits per month; pay-as-you-go $0.008 per credit; paid plans from $30/month | [Tavily API credits](https://docs.tavily.com/documentation/api-credits), checked 2026-10-27. The app uses basic depth, ≤ 4 results and ≤ 2 searches per worker task, so a question costs 0–4 credits. The plan of the key used here was not verified |
| `text-embedding-3-small` | $0.02 (input only) | OpenAI model page, checked 2026-10-26 |

Access: an OpenAI API key, an OpenRouter API key and a Tavily API key. In this deployment, OpenRouter calls Google with the student's own key (BYOK), so OpenRouter reports $0 and the upstream cost is billed by Google. Each call's cost is shown in the Routing tab: OpenRouter's reported cost for Gemini calls ("reported"), and an estimate from the dated price table for OpenAI calls ("est."), because OpenAI returns no per-call cost. After the P4-D1 price fix, a typical question measured on 2026-10-27 cost about $0.02–0.04. Cloud Run costs are covered by GCP credits; LLM APIs bill separately. Full cost and latency numbers: Phase 9 (`docs/evaluation.md`).
