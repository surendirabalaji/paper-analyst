"""Runtime settings, read once from the environment (and `.env` locally).

Credentials stay in the environment; nothing here is logged or sent to the browser.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def _path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (ROOT / p).resolve()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    openrouter_api_key: str
    openrouter_base_url: str
    gemini_model: str
    embedding_model: str
    chroma_dir: Path
    chroma_collection: str
    top_k: int
    max_supervisor_steps: int
    max_tool_rounds: int
    worker_timeout_s: float
    request_budget_s: float
    llm_timeout_s: float
    # Routing (Phase 4). Defaults keep the dataclass constructible in tests.
    gemini_reasoning_effort: str = "low"  # OpenRouter reasoning effort; Gemini 3.8 cannot disable reasoning
    gemini_planner_reasoning_effort: str = ""  # planner only; empty = same as gemini_reasoning_effort
    openai_reasoning_effort: str = ""  # empty = model default (the capability side keeps its reasoning)
    router_force_provider: str = ""  # evaluation-only override: "openai" or "gemini-openrouter"
    fallback_enabled: bool = True
    # Live retrieval (Phase 5): Tavily Remote MCP, key sent only as the tavilyApiKey query parameter.
    tavily_api_key: str = ""
    tavily_mcp_base_url: str = "https://mcp.tavily.com/mcp/"
    tavily_timeout_s: float = 15.0
    tavily_max_results: int = 4
    rag_min_score: float = 0.55  # best corpus similarity below this = low corpus support (calibrated 2026-10-27)
    sqlite_path: Path = ROOT / "data" / "papers.db"
    # Failure handling (Phase 7). The synthesis reserve is request time kept back from workers so the
    # final answer can still stream inside REQUEST_BUDGET_S.
    synth_reserve_s: float = 30.0
    fault_inject: str = ""  # process-wide fault spec, local runs only (app/faults.py)
    fault_inject_token: str = ""  # enables the per-request X-Fault-Inject header when it matches X-Fault-Token
    # Latency (Phase 8b, package B). Off in the dataclass so tests opt in; on by default in load_settings.
    planner_seeds: bool = False  # run the planner-named papers / queries as the worker's first tool calls
    evidence_only_single_task: bool = False  # one-task plans: the worker gathers evidence, synthesis writes
    speculative_retrieval: bool = False  # corpus search of the question runs while the planner plans


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env", override=False)
    env = os.environ.get
    return Settings(
        openai_api_key=env("OPENAI_API_KEY", ""),
        openai_model=env("OPENAI_MODEL", "gpt-6.1-sol"),
        openrouter_api_key=env("OPENROUTER_API_KEY", ""),
        openrouter_base_url=env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        gemini_model=env("GEMINI_MODEL", "google/gemini-3.8-flash"),
        embedding_model=env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        chroma_dir=_path(env("CHROMA_PERSIST_DIR", "./data/chroma_index")),
        chroma_collection=env("CHROMA_COLLECTION", "arxiv_agents"),
        top_k=int(env("RAG_TOP_K", "6")),
        max_supervisor_steps=int(env("MAX_SUPERVISOR_STEPS", "6")),
        max_tool_rounds=int(env("MAX_TOOL_ROUNDS", "4")),
        worker_timeout_s=float(env("WORKER_TIMEOUT_S", "45")),
        request_budget_s=float(env("REQUEST_BUDGET_S", "120")),
        llm_timeout_s=float(env("LLM_TIMEOUT_S", "60")),
        gemini_reasoning_effort=env("GEMINI_REASONING_EFFORT", "minimal"),
        gemini_planner_reasoning_effort=env("GEMINI_PLANNER_REASONING_EFFORT", "low"),
        openai_reasoning_effort=env("OPENAI_REASONING_EFFORT", ""),
        router_force_provider=env("ROUTER_FORCE_PROVIDER", ""),
        fallback_enabled=env("LLM_FALLBACK", "1") != "0",
        tavily_api_key=env("TAVILY_API_KEY", ""),
        tavily_mcp_base_url=env("TAVILY_MCP_BASE_URL", "https://mcp.tavily.com/mcp/"),
        tavily_timeout_s=float(env("TAVILY_TIMEOUT_S", "15")),
        tavily_max_results=int(env("TAVILY_MAX_RESULTS", "4")),
        rag_min_score=float(env("RAG_MIN_SCORE", "0.55")),
        sqlite_path=_path(env("SQLITE_PATH", "./data/papers.db")),
        synth_reserve_s=float(env("SYNTH_RESERVE_S", "30")),
        fault_inject=env("FAULT_INJECT", ""),
        fault_inject_token=env("FAULT_INJECT_TOKEN", ""),
        planner_seeds=env("PLANNER_SEEDS", "1") != "0",
        evidence_only_single_task=env("EVIDENCE_ONLY_SINGLE_TASK", "1") != "0",
        speculative_retrieval=env("SPECULATIVE_RETRIEVAL", "1") != "0",
    )
