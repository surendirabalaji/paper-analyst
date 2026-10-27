"""Chroma vector index and SQLite metadata DB writers.

Chroma: one collection, cosine HNSW, no Chroma-side embedding function (embeddings come from
OpenAI and are passed explicitly). The embedding model and dimensions are stored in the
collection metadata; opening the index with a different model is refused unless the caller
asks for a rebuild.

Idempotency: chunk IDs are deterministic. For each paper the writer compares the new chunk set
with what the index holds (by chunk_id and content_sha256) and only embeds new or changed
chunks; stale chunks of that paper (older versions, chunks a parser change removed) are deleted.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import chromadb

from ingest.chunking import Chunk

CHROMA_SETTINGS = chromadb.Settings(anonymized_telemetry=False, allow_reset=False)


class EmbeddingMismatch(RuntimeError):
    pass


def open_client(persist_dir: str | Path) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(persist_dir), settings=CHROMA_SETTINGS)


def open_collection(
    client: chromadb.ClientAPI,
    name: str,
    embedding_model: str,
    dimensions: int | None,
    *,
    rebuild: bool = False,
):
    """Open or create the collection. dimensions=None opens an existing collection and checks only
    the model name (its stored dimensions are then authoritative); creating requires dimensions."""
    existing = {c.name for c in client.list_collections()}
    if name in existing:
        col = client.get_collection(name)
        meta = col.metadata or {}
        same = meta.get("embedding_model") == embedding_model and (
            dimensions is None or int(meta.get("embedding_dimensions", 0)) == dimensions
        )
        if not same and not rebuild:
            raise EmbeddingMismatch(
                f"collection {name!r} was built with {meta.get('embedding_model')}/"
                f"{meta.get('embedding_dimensions')}, configured {embedding_model}/{dimensions}; "
                "re-run ingest with --rebuild"
            )
        if not rebuild:
            return col
        client.delete_collection(name)
    if dimensions is None:
        raise ValueError("embedding dimensions are required to create a collection")
    return client.create_collection(
        name,
        embedding_function=None,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": embedding_model,
            "embedding_dimensions": dimensions,
        },
    )


@dataclass
class UpsertStats:
    added: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0
    embedded_tokens: int = 0
    deleted_ids: list[str] = field(default_factory=list)


def sync_paper_chunks(
    col,
    arxiv_id: str,
    chunks: list[Chunk],
    embed: Callable[[list[str]], list[list[float]]],
    batch: int = 100,
) -> UpsertStats:
    """Make the index hold exactly `chunks` for this paper (all versions)."""
    stats = UpsertStats()
    have = col.get(where={"arxiv_id": arxiv_id}, include=["metadatas"])
    have_sha = {i: (m or {}).get("content_sha256") for i, m in zip(have["ids"], have["metadatas"])}
    want = {c.chunk_id: c for c in chunks}
    if len(want) != len(chunks):
        raise ValueError(f"duplicate chunk ids for {arxiv_id}")

    stale = [i for i in have_sha if i not in want]
    if stale:
        col.delete(ids=stale)
        stats.deleted = len(stale)
        stats.deleted_ids = stale

    todo: list[Chunk] = []
    for cid, c in want.items():
        if cid not in have_sha:
            stats.added += 1
            todo.append(c)
        elif have_sha[cid] != c.sha256:
            stats.updated += 1
            todo.append(c)
        else:
            stats.unchanged += 1

    for i in range(0, len(todo), batch):
        part = todo[i : i + batch]
        vectors = embed([c.text for c in part])
        col.upsert(
            ids=[c.chunk_id for c in part],
            embeddings=vectors,
            documents=[c.text for c in part],
            metadatas=[c.metadata for c in part],
        )
        stats.embedded_tokens += sum(c.metadata["n_tokens"] for c in part)
    return stats


def prune_papers(col, keep_arxiv_ids: set[str]) -> int:
    """Delete chunks of papers no longer in the corpus selection."""
    got = col.get(include=["metadatas"])
    drop = [i for i, m in zip(got["ids"], got["metadatas"]) if (m or {}).get("arxiv_id") not in keep_arxiv_ids]
    if drop:
        col.delete(ids=drop)
    return len(drop)


# ---------------------------------------------------------------- SQLite

SCHEMA = """
CREATE TABLE papers (
    arxiv_id         TEXT PRIMARY KEY,           -- base id, e.g. 2310.11511
    version          INTEGER NOT NULL,
    doc_id           TEXT NOT NULL UNIQUE,       -- versioned id used in chunk ids, e.g. 2310.11511v1
    title            TEXT NOT NULL,
    abstract         TEXT NOT NULL,
    published        TEXT NOT NULL,              -- v1 submission date, YYYY-MM-DD
    updated          TEXT NOT NULL,              -- date of the indexed version
    year             INTEGER NOT NULL,
    primary_category TEXT NOT NULL,
    comment          TEXT,
    journal_ref      TEXT,
    doi              TEXT,
    abs_url          TEXT NOT NULL,
    html_url         TEXT NOT NULL,
    fulltext         INTEGER NOT NULL,           -- 1 if chunks are in the vector index
    word_count       INTEGER NOT NULL DEFAULT 0,
    n_sections       INTEGER NOT NULL DEFAULT 0,
    n_chunks         INTEGER NOT NULL DEFAULT 0,
    topic            TEXT NOT NULL               -- curation group from data/corpus_seed.json
);
CREATE TABLE authors (
    author_id INTEGER PRIMARY KEY,
    name      TEXT NOT NULL UNIQUE
);
CREATE TABLE paper_authors (
    arxiv_id  TEXT NOT NULL REFERENCES papers(arxiv_id),
    author_id INTEGER NOT NULL REFERENCES authors(author_id),
    position  INTEGER NOT NULL,                  -- 1 = first author
    PRIMARY KEY (arxiv_id, author_id)
);
CREATE TABLE paper_categories (
    arxiv_id   TEXT NOT NULL REFERENCES papers(arxiv_id),
    category   TEXT NOT NULL,
    is_primary INTEGER NOT NULL,
    PRIMARY KEY (arxiv_id, category)
);
CREATE TABLE repos (
    arxiv_id TEXT PRIMARY KEY REFERENCES papers(arxiv_id),
    owner    TEXT NOT NULL,
    name     TEXT NOT NULL,
    url      TEXT NOT NULL,
    found_in TEXT NOT NULL                       -- comment | abstract | html_front_matter
);
CREATE INDEX idx_paper_authors_author ON paper_authors(author_id);
CREATE INDEX idx_paper_categories_cat ON paper_categories(category);
"""


def write_sqlite(path: str | Path, papers: list[dict]) -> dict:
    """Rebuild the metadata DB from scratch (atomic replace). Returns table row counts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()
    con = sqlite3.connect(tmp)
    try:
        con.executescript(SCHEMA)
        author_ids: dict[str, int] = {}
        for p in sorted(papers, key=lambda x: x["arxiv_id"]):
            con.execute(
                "INSERT INTO papers VALUES (:arxiv_id,:version,:doc_id,:title,:abstract,:published,:updated,"
                ":year,:primary_category,:comment,:journal_ref,:doi,:abs_url,:html_url,:fulltext,"
                ":word_count,:n_sections,:n_chunks,:topic)",
                {**p, "fulltext": int(p["fulltext"]), "year": int(p["published"][:4])},
            )
            seen: set[int] = set()
            for pos, name in enumerate(p["authors"], start=1):
                if name not in author_ids:
                    author_ids[name] = con.execute("INSERT INTO authors(name) VALUES (?)", (name,)).lastrowid
                aid = author_ids[name]
                if aid in seen:  # same name listed twice in arXiv metadata
                    continue
                seen.add(aid)
                con.execute("INSERT INTO paper_authors VALUES (?,?,?)", (p["arxiv_id"], aid, pos))
            for cat in dict.fromkeys(p["categories"]):
                con.execute(
                    "INSERT INTO paper_categories VALUES (?,?,?)",
                    (p["arxiv_id"], cat, int(cat == p["primary_category"])),
                )
            if p.get("repo"):
                r = p["repo"]
                con.execute(
                    "INSERT INTO repos VALUES (?,?,?,?,?)",
                    (p["arxiv_id"], r["owner"], r["name"], r["url"], r["found_in"]),
                )
        con.commit()
        counts = {
            t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("papers", "authors", "paper_authors", "paper_categories", "repos")
        }
    finally:
        con.close()
    os.replace(tmp, path)
    return counts
