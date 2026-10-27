"""T0-IMP: the pinned frameworks import and basic offline operations work on Python 3.13."""
import importlib
import sys

import pytest

MODULES = [
    "langgraph.graph",
    "langchain_core.runnables",
    "langchain_openai",
    "langchain_chroma",
    "langchain_text_splitters",
    "langchain_mcp_adapters.client",
    "mcp.client.streamable_http",
    "chromadb",
    "fastapi",
    "uvicorn",
    "pydantic_settings",
    "httpx",
    "tiktoken",
    "bs4",
    "lxml.etree",
    "arxiv",
    "ragas",
    "dotenv",
]


def test_python_313():
    assert sys.version_info[:2] == (3, 13)


@pytest.mark.parametrize("mod", MODULES)
def test_import(mod):
    importlib.import_module(mod)


def test_ragas_four_metrics_importable():
    # Legacy names (deprecated in ragas 0.4.x, slated for removal in 1.0).
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import (  # noqa: F401
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    # Current collections API names in the pinned version.
    from ragas.metrics.collections import (  # noqa: F401
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
    )
    from ragas.metrics.collections import Faithfulness as F2  # noqa: F401


def test_langgraph_compiles_and_runs_offline():
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class S(TypedDict):
        n: int

    g = StateGraph(S)
    g.add_node("inc", lambda s: {"n": s["n"] + 1})
    g.add_edge(START, "inc")
    g.add_edge("inc", END)
    assert g.compile().invoke({"n": 1}) == {"n": 2}


def test_chroma_ephemeral_vector_query_offline():
    import chromadb

    c = chromadb.EphemeralClient()
    col = c.create_collection("phase0_probe", metadata={"hnsw:space": "cosine"})
    col.add(ids=["a", "b"], embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], documents=["x", "y"])
    res = col.query(query_embeddings=[[0.9, 0.1, 0.0]], n_results=1)
    assert res["ids"][0] == ["a"]


def test_streamable_http_transport_available_in_adapter():
    # The adapter must accept the streamable_http transport used for Tavily Remote MCP.
    from langchain_mcp_adapters.client import MultiServerMCPClient

    MultiServerMCPClient({"t": {"transport": "streamable_http", "url": "https://example.invalid/mcp/"}})
