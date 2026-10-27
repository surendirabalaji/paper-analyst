"""Smoke-test the built index: top-k vector search through langchain-chroma.

Usage: .venv/bin/python scripts/query_index.py "how does Self-RAG decide when to retrieve?" [-k 6]
Prints chunk ids, cosine relevance scores, section paths and anchor URLs. Costs one embedding call.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402
from langchain_openai import OpenAIEmbeddings  # noqa: E402

from ingest.store import open_client  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=int(os.environ.get("RAG_TOP_K", 6)))
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    persist = ROOT / os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma_index")
    name = os.environ.get("CHROMA_COLLECTION", "arxiv_agents")

    client = open_client(persist)
    meta = client.get_collection(name).metadata or {}
    if meta.get("embedding_model") != model:
        print(f"Index built with {meta.get('embedding_model')}, configured {model}: rebuild needed")
        return 2
    emb = OpenAIEmbeddings(model=model)
    dims = len(emb.embed_query(args.query))
    if int(meta.get("embedding_dimensions", 0)) != dims:
        print(f"Index has {meta.get('embedding_dimensions')}-d vectors, {model} returns {dims}-d: rebuild needed")
        return 2
    store = Chroma(client=client, collection_name=name, embedding_function=emb)
    for doc, score in store.similarity_search_with_relevance_scores(args.query, k=args.k):
        md = doc.metadata
        cid = f"{md['doc_id']}::{md['section_id']}::{md['chunk_index']:03d}"
        print(f"{score:.3f}  {cid}\n       {md['section_path']}\n       {md['html_anchor_url']}")
        print("       " + doc.page_content.split("\n", 1)[1][:160].replace("\n", " ") + " ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
