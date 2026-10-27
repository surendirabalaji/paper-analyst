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
| Generation (provider 1) | OpenAI API (direct) | `gpt-6.1-sol` *(candidate, to verify)* |
| Generation (provider 2) | Google Gemini via OpenRouter | `google/gemini-3.8-flash` *(candidate, to verify)* |
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

*TBD (Phase 1)*

## Run locally

*TBD (Phase 2)*

## Deploy to a fresh GCP project

*TBD (Phase 3)*

## Evaluation

*TBD (Phase 9)*

## Cost and access (dated)

*TBD.* Prices are verified on each provider's pricing page and dated here before submission.
