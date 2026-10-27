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
