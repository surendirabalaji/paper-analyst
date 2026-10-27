"""Phase 1 integration against the REAL built corpus (data/chroma_index, data/papers.db, data/corpus_manifest.json).

Read-only checks run by default when the artifacts exist (skip otherwise; a skip is not a pass).
- RUN_LIVE=1: real vector retrieval (one OpenAI embedding per query).
- RUN_REAL_INGEST=1: re-runs scripts/ingest.py on the real index (no flags; expects 0 changes, 0 embeddings,
  0 arXiv requests). Rewrites data/corpus_manifest.json timestamps.

IDs: T1-REAL-*, T1-LIVE-*.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys

import pytest

from ingest.chunking import CHUNK_TOKENS, n_tokens

REQ_META = ("arxiv_id", "doc_id", "title", "section_id", "section_title", "section_path", "chunk_index",
            "html_anchor_url", "content_sha256", "n_tokens")
ID_RE = re.compile(r"^(\d{4}\.\d{4,5})v(\d+)::([A-Za-z0-9.]+)::(\d{3})$")


@pytest.fixture(scope="module")
def real(root):
    idx, db, man = root / "data/chroma_index", root / "data/papers.db", root / "data/corpus_manifest.json"
    if not (idx.exists() and db.exists() and man.exists()):
        pytest.skip("real index/DB not built")
    import chromadb

    client = chromadb.PersistentClient(path=str(idx), settings=chromadb.Settings(anonymized_telemetry=False))
    manifest = json.loads(man.read_text())
    col = client.get_collection(manifest["vector_store"]["collection"])
    got = col.get(include=["metadatas", "documents"])
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    return {"col": col, "got": got, "db": con, "manifest": manifest, "root": root}


def test_T1_REAL_counts_consistent(real):
    m, col, db = real["manifest"], real["col"], real["db"]
    t = m["totals"]
    assert 60 <= t["fulltext_indexed"] <= 100, "agreed target ~80 full-text papers (60-100)"
    assert t["papers"] == t["fulltext_indexed"] + t["metadata_only"] == len(m["papers"])
    assert col.count() == t["chunks"] == m["vector_store"]["count"] == sum(p["n_chunks"] for p in m["papers"])
    assert db.execute("SELECT COUNT(*), SUM(fulltext), SUM(n_chunks) FROM papers").fetchone() == (t["papers"], t["fulltext_indexed"], t["chunks"])
    per_doc = collections.Counter(md["doc_id"] for md in real["got"]["metadatas"])
    for p in m["papers"]:
        assert per_doc.get(p["doc_id"], 0) == p["n_chunks"], p["doc_id"]
        assert (p["n_chunks"] > 0) == p["fulltext"] == (p["status"] == "indexed")
    seed = json.loads((real["root"] / "data/corpus_seed.json").read_text())
    assert sorted(p["arxiv_id"] for p in seed["papers"]) == sorted(p["arxiv_id"] for p in m["papers"])
    assert len({p["topic"] for p in m["papers"]}) >= 5, "topic breadth"


def test_T1_REAL_chunk_ids_and_metadata(real):
    got = real["got"]
    assert len(set(got["ids"])) == len(got["ids"])
    for cid, md, doc in zip(got["ids"], got["metadatas"], got["documents"]):
        for k in REQ_META:
            assert md.get(k) not in (None, ""), (cid, k)
        mm = ID_RE.match(cid)
        assert mm, cid
        assert cid == f"{md['doc_id']}::{md['section_id']}::{md['chunk_index']:03d}"
        assert md["doc_id"] == f"{md['arxiv_id']}v{mm.group(2)}"
        assert hashlib.sha256(doc.encode()).hexdigest() == md["content_sha256"], cid
        anchor = f"https://arxiv.org/html/{md['doc_id']}" + ("" if md["section_id"] == "abstract" else f"#{md['section_id']}")
        assert md["html_anchor_url"] == anchor
        assert doc.startswith(f"{md['title']} — {md['section_path']}\n"), cid


def test_T1_REAL_chunk_versions_match_manifest_and_db(real):
    pinned = {p["arxiv_id"]: p["doc_id"] for p in real["manifest"]["papers"]}
    db = dict(real["db"].execute("SELECT arxiv_id, doc_id FROM papers"))
    assert db == pinned
    assert all(pinned[md["arxiv_id"]] == md["doc_id"] for md in real["got"]["metadatas"])


def test_T1_REAL_chunk_token_bounds(real):
    for md, doc in zip(real["got"]["metadatas"], real["got"]["documents"]):
        body = doc.split("\n", 1)[1]
        assert n_tokens(body) <= CHUNK_TOKENS
        assert md["n_tokens"] == n_tokens(doc)


def test_T1_REAL_anchors_exist_in_cached_html_and_version_in_page(real):
    raw = real["root"] / "data/raw"
    if not raw.exists():
        pytest.skip("raw HTML cache absent")
    cache, missing = {}, []
    for md in real["got"]["metadatas"]:
        d = md["doc_id"]
        if d not in cache:
            cache[d] = (raw / f"{d}.html").read_text()
            assert d in cache[d], f"in-page version mismatch for {d}"
        if md["section_id"] != "abstract" and f'id="{md["section_id"]}"' not in cache[d]:
            missing.append(md["section_id"])
    assert missing == []


def test_T1_REAL_no_parser_noise(real):
    bad = collections.Counter()
    for doc in real["got"]["documents"]:
        for pat in ("⬇", "\\lxSVG", "\\pgfpicture", "Report issue for preceding element", "ltx_"):
            if pat in doc:
                bad[pat] += 1
    assert not bad, bad


def test_T1_REAL_no_acknowledgment_chunks(real):
    """P1-D3 regression: no chunk from an acknowledgment/author-contribution section or run-in ack paragraph."""
    ack_title = re.compile(r"(?i)^(acknowledg|author contributions?\b)")
    thanks = re.compile(r"(?i)\bwe (would like to )?thank\b|\backnowledge?ments?[.:]")
    bad = []
    for cid, doc, md in zip(real["got"]["ids"], real["got"]["documents"], real["got"]["metadatas"]):
        if any(ack_title.match(part.strip()) for part in md["section_path"].split("›")) or thanks.search(doc):
            bad.append(cid)
    assert not bad, bad[:10]


def test_T1_REAL_embeddings_dims_and_model(real):
    md = real["col"].metadata
    emb = real["manifest"]["embedding"]
    assert md["embedding_model"] == emb["model"] == os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    assert int(md["embedding_dimensions"]) == emb["dimensions"] == 1536
    sample = real["got"]["ids"][:: max(1, len(real["got"]["ids"]) // 50)]
    vecs = real["col"].get(ids=sample, include=["embeddings"])["embeddings"]
    import numpy as np

    for v in vecs:
        v = np.asarray(v)
        assert v.shape == (1536,) and abs(np.linalg.norm(v) - 1) < 1e-2 and np.count_nonzero(v) > 1000


def test_T1_REAL_sqlite_matches_independent_truth(real):
    raw = real["root"] / "data/raw/arxiv_metadata.json"
    if not raw.exists():
        pytest.skip("metadata cache absent")
    meta = json.loads(raw.read_text())
    db = real["db"]
    ids = [r[0] for r in db.execute("SELECT arxiv_id FROM papers")]
    authors = collections.defaultdict(list)
    for a, n in db.execute("SELECT pa.arxiv_id, au.name FROM paper_authors pa JOIN authors au USING(author_id) ORDER BY pa.arxiv_id, pa.position"):
        authors[a].append(n)
    for a in ids:
        assert authors[a] == list(dict.fromkeys(meta[a]["authors"])), a
    assert db.execute("SELECT COUNT(*) FROM authors").fetchone()[0] == len({n for a in ids for n in meta[a]["authors"]})
    cats = collections.defaultdict(set)
    for a, c, p in db.execute("SELECT arxiv_id, category, is_primary FROM paper_categories"):
        cats[a].add((c, p))
    for a in ids:
        assert cats[a] == {(c, int(c == meta[a]["primary_category"])) for c in meta[a]["categories"]}, a
    man = {p["arxiv_id"]: p for p in real["manifest"]["papers"]}
    repos = {r[0]: r[1:] for r in db.execute("SELECT arxiv_id, owner, name, url, found_in FROM repos")}
    assert set(repos) == {a for a, p in man.items() if p["repo"]}
    for a, (o, n, u, f) in repos.items():
        assert man[a]["repo"] == {"owner": o, "name": n, "url": u, "found_in": f}
        assert u == f"https://github.com/{o}/{n}" and not n.endswith((".git", "."))
    assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    # year filter sanity (Data agent will rely on it)
    assert db.execute("SELECT COUNT(*) FROM papers WHERE year <> CAST(substr(published,1,4) AS INT)").fetchone() == (0,)


def test_T1_REAL_generated_artifacts_ignored(root):
    r = subprocess.run(["git", "check-ignore", "data/raw", "data/chroma_index", "data/papers.db", "data/papers.tmp"],
                       cwd=root, capture_output=True, text=True)
    assert len(r.stdout.split()) == 4
    tracked = subprocess.run(["git", "ls-files", "data"], cwd=root, capture_output=True, text=True).stdout.split()
    assert not any(t.startswith(("data/raw", "data/chroma_index")) or t.endswith(".db") for t in tracked)


# ------------------------------------------------------------------ live (paid, tiny)

def _search(real, q, k):
    """Same code path as scripts/query_index.py: langchain-chroma vector store over the persisted collection."""
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    from ingest.store import open_client

    client = open_client(real["root"] / "data/chroma_index")
    store = Chroma(client=client, collection_name=real["col"].name,
                   embedding_function=OpenAIEmbeddings(model=real["col"].metadata["embedding_model"]))
    return store.similarity_search_with_relevance_scores(q, k=k)


@pytest.mark.live
def test_T1_LIVE_selfrag_retrieval(real):
    from dotenv import load_dotenv

    load_dotenv(real["root"] / ".env")
    res = _search(real, "how does Self-RAG decide when to retrieve?", 5)
    ids = [d.metadata["arxiv_id"] for d, _ in res]
    assert ids.count("2310.11511") == 5
    secs = {d.metadata["section_id"] for d, _ in res}
    assert secs & {"S3.SS3", "A1.SS3"}, secs  # "Self-Rag Inference" sections
    for d, _ in res:
        assert d.metadata["html_anchor_url"].startswith("https://arxiv.org/html/2310.11511v1")


@pytest.mark.live
@pytest.mark.parametrize("q,target,k,need", [
    ("what failure categories cause teams of cooperating LLM agents to break down?", "2503.13657", 3, 3),
    ("what signal tells an LM trained with reflection tokens to fetch passages on demand?", "2310.11511", 8, 3),
])
def test_T1_LIVE_low_overlap_paraphrase(real, q, target, k, need):
    from dotenv import load_dotenv

    load_dotenv(real["root"] / ".env")
    res = _search(real, q, k)
    assert sum(d.metadata["arxiv_id"] == target for d, _ in res) >= need, [d.metadata["doc_id"] for d, _ in res]


@pytest.mark.skipif(os.environ.get("RUN_REAL_INGEST") != "1", reason="set RUN_REAL_INGEST=1 to re-run ingest on the real index")
def test_T1_REAL_rerun_ingest_adds_zero(root):
    before = json.loads((root / "data/corpus_manifest.json").read_text())
    r = subprocess.run([sys.executable, str(root / "scripts/ingest.py")], cwd=root, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "arXiv requests: 0" in r.stdout
    after = json.loads((root / "data/corpus_manifest.json").read_text())
    lr = after["last_run"]
    assert (lr["added"], lr["updated"], lr["deleted"], lr["embedded_tokens"], lr["pruned_other_papers"]) == (0, 0, 0, 0, 0)
    assert lr["chunks_before"] == lr["chunks_after"] == before["totals"]["chunks"]
    strip = lambda m: {k: v for k, v in m.items() if k not in ("generated_at", "last_run")}  # noqa: E731
    assert strip(before) == strip(after)
