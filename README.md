# Agentic AI Paper Analyst

A multi-agent system for CSCI 599 Assignment 2. It answers questions about a corpus of arXiv papers on LLM agents and RAG. A LangGraph supervisor delegates to specialized agents (Research, Summary, Data, and Code as a stretch goal). They use RAG over a Chroma index, live web search through Tavily Remote MCP, and generation routed between two providers (OpenAI and Gemini via OpenRouter). Answers stream to a vanilla HTML/JS page over SSE, with the routing decisions, an orchestration trace and clickable citations.

> **Status: under construction (phase-wise build).** Sections marked *TBD* are filled in as phases complete.

- **Live URL:** *TBD*
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

## Deploy to a fresh GCP project

*TBD (Phase 3)*

## Evaluation

*TBD (Phase 9)*

## Cost and access (dated)

*TBD.* Prices are verified on each provider's pricing page and dated here before submission.
