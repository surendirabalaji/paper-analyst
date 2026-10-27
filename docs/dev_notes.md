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
- **Bug found: ragas import failure.** `import ragas` (0.4.3, latest) failed: it imports `langchain_community.chat_models.vertexai`, which `langchain-community` 0.4.2 removed. A clean install did not catch this; only an import test did. Fix: hold `langchain-community==0.4.1` (still has the module, accepts langchain-core 1.x). Lockfile recompiled: 148 exact pins. All four RAGAS metric classes import (`Faithfulness`, `ResponseRelevancy`, `LLMContextPrecisionWithReference`, `LLMContextRecall`).
- **Model IDs (OpenRouter public catalog, 2026-10-26):** `google/gemini-3.8-flash` listed, tools supported, $0.825 in / $4.125 out per 1M tokens. `openai/gpt-6.1-sol` listed at $2 / $10 (OpenRouter price; OpenAI direct price still to confirm). Cost ordering assumed by the router holds (Gemini cheaper).
- **arXiv HTML samples** (2210.03629 ReAct, 2310.11511 Self-RAG, 2401.15884 CRAG): all three return HTML, including a 2022 paper. LaTeXML gives stable IDs (`S3`, `S3.SS2`, appendix `A1.SS1`), `<math alttext>`, figcaptions, and a separate `ltx_bibliography`. 7.6k–18.7k words excluding the bibliography. Parser issues for Phase 1: `get_text(" ")` fragments small-caps headings ("Re asoning"); the HTML URL has no version, so take the version from the arXiv API.
- `scripts/phase0_check.py` written for the keyed services; waiting on `.env`.
