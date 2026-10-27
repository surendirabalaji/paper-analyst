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
    default_provider: str  # Phase 2: every call uses this provider; the router arrives in Phase 4
    embedding_model: str
    chroma_dir: Path
    chroma_collection: str
    top_k: int
    max_supervisor_steps: int
    max_tool_rounds: int
    worker_timeout_s: float
    request_budget_s: float
    llm_timeout_s: float


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env", override=False)
    env = os.environ.get
    return Settings(
        openai_api_key=env("OPENAI_API_KEY", ""),
        openai_model=env("OPENAI_MODEL", "gpt-6.1-sol"),
        openrouter_api_key=env("OPENROUTER_API_KEY", ""),
        openrouter_base_url=env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        gemini_model=env("GEMINI_MODEL", "google/gemini-3.8-flash"),
        default_provider=env("DEFAULT_PROVIDER", "openai"),
        embedding_model=env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        chroma_dir=_path(env("CHROMA_PERSIST_DIR", "./data/chroma_index")),
        chroma_collection=env("CHROMA_COLLECTION", "arxiv_agents"),
        top_k=int(env("RAG_TOP_K", "6")),
        max_supervisor_steps=int(env("MAX_SUPERVISOR_STEPS", "6")),
        max_tool_rounds=int(env("MAX_TOOL_ROUNDS", "4")),
        worker_timeout_s=float(env("WORKER_TIMEOUT_S", "45")),
        request_budget_s=float(env("REQUEST_BUDGET_S", "120")),
        llm_timeout_s=float(env("LLM_TIMEOUT_S", "60")),
    )
