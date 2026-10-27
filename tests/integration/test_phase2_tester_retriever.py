"""Tester Phase 2 (T2-R): Retriever guards against the REAL index (no embedding API calls; the embedding
client is replaced only where a query vector is needed)."""
import asyncio
import dataclasses
import pathlib

import pytest

from app.config import load_settings
from app.rag import Retriever
from ingest.store import EmbeddingMismatch

ROOT = pathlib.Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not (ROOT / "data/chroma_index").exists(), reason="real index not built")


def test_T2_R1_startup_refuses_embedding_model_mismatch():
    s = dataclasses.replace(load_settings(), embedding_model="text-embedding-3-large")
    with pytest.raises(EmbeddingMismatch):
        Retriever(s)


def test_T2_R2_query_dimension_mismatch_is_refused():
    r = Retriever(load_settings())

    class Wrong:
        async def aembed_query(self, q):
            return [0.0] * 3072

    r.embeddings = Wrong()
    with pytest.raises(EmbeddingMismatch):
        asyncio.run(r.search("anything", 6))


def test_T2_R3_vector_path_and_k_clamp_with_stored_vector():
    """Use a stored chunk's own vector as the query: top-1 must be that chunk (proves vector similarity
    retrieval over real embeddings), and k is clamped to [3, 10]."""
    from ingest.store import open_client

    r = Retriever(load_settings())
    col = open_client(ROOT / "data/chroma_index").get_collection("arxiv_agents")
    got = col.get(ids=["2305.18323v1::S1::000"], include=["embeddings"])
    vec = list(got["embeddings"][0])

    class Fixed:
        async def aembed_query(self, q):
            return vec

    r.embeddings = Fixed()
    top = asyncio.run(r.search("ignored", 1))
    assert len(top) == 3 and top[0].chunk_id == "2305.18323v1::S1::000" and top[0].score > 0.99
    assert len(asyncio.run(r.search("ignored", 50))) == 10
