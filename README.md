# Agentic AI Paper Analyst

A multi-agent system for CSCI 599 Assignment 2. It answers questions about a corpus of arXiv papers on LLM agents and RAG. A LangGraph supervisor delegates to specialized agents (Research, Summary, Data, and Code as a stretch goal). They use RAG over a Chroma index, live web search through Tavily Remote MCP, and generation routed between two providers (OpenAI and Gemini via OpenRouter). Answers stream to a vanilla HTML/JS page over SSE, with the routing decisions, an orchestration trace and clickable citations.

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
| Live web retrieval | Tavily Remote MCP | `https://mcp.tavily.com/mcp/?tavilyApiKey=…` |
| Vector store | Chroma (embedded, baked into the image) | |
| Hosting | Google Cloud Run | |

## Local setup

```bash
uv python install 3.13
uv venv -p 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env   # then fill in the keys
```

Dependencies: `requirements.in` lists the top-level packages; `requirements.txt` pins every package with `==` and is generated with:

```bash
uv pip compile requirements.in --python-version 3.13 \
  --python-platform x86_64-unknown-linux-gnu -o requirements.txt
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

Needs the built index (`data/chroma_index`, see above) and `.env` with at least `OPENAI_API_KEY`.

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

Prerequisites: the [gcloud CLI](https://cloud.google.com/sdk/docs/install), a GCP project with billing enabled (or credits), the built index in `data/` (see *Corpus ingestion*), and a filled-in `.env` (at least `OPENAI_API_KEY`; `OPENROUTER_API_KEY` and `TAVILY_API_KEY` are used from Phases 4 and 5).

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

Credentials end up as Cloud Run environment variables on the service (visible to project members with Cloud Run access), not in the image or the repo. To rotate a key, edit `.env` and rerun the script.

Time a deployed instance from the client (health, TTFT, total): `.venv/bin/python scripts/probe_remote.py https://YOUR-SERVICE-URL`.

## Evaluation

*TBD (Phase 9)*

## Cost and access (dated)

*TBD.* Prices are verified on each provider's pricing page and dated here before submission.
