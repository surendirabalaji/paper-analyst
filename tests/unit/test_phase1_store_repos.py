"""Phase 1 unit tests: Chroma sync/idempotency/version cleanup/mismatch refusal, SQLite writer,
repo-link normalisation, arXiv Atom parsing. Temporary dirs and fake embeddings only.

IDs: T1-STORE-*, T1-SQL-*, T1-REPO-*, T1-ATOM-*.
"""
from __future__ import annotations

import hashlib
import sqlite3

import pytest

from ingest.arxiv_client import PaperMeta, parse_atom, split_version
from ingest.chunking import Chunk
from ingest.repos import _normalize, find_repo
from ingest.store import EmbeddingMismatch, open_client, open_collection, prune_papers, sync_paper_chunks, write_sqlite

DIMS = 8


class FakeEmbed:
    def __init__(self, dims=DIMS):
        self.calls, self.texts, self.dims = 0, 0, dims

    def __call__(self, texts):
        self.calls += 1
        self.texts += len(texts)
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            out.append([b / 255 + 0.01 for b in h[: self.dims]])
        return out


def mk(arxiv_id, version, sid, n, text):
    doc = f"{arxiv_id}v{version}"
    return Chunk(
        chunk_id=f"{doc}::{sid}::{n:03d}",
        text=text,
        metadata={"arxiv_id": arxiv_id, "doc_id": doc, "section_id": sid, "section_title": sid, "chunk_index": n,
                  "html_anchor_url": f"https://arxiv.org/html/{doc}#{sid}", "n_tokens": len(text.split()),
                  "content_sha256": hashlib.sha256(text.encode()).hexdigest()},
    )


@pytest.fixture()
def col(tmp_path):
    return open_collection(open_client(tmp_path / "chroma"), "tcol", "fake-model", DIMS)


def paper(aid="1111.00001", v=1, texts=("alpha text", "beta text", "gamma text")):
    return [mk(aid, v, "S1", i, t) for i, t in enumerate(texts)]


def test_T1_STORE_collection_metadata(col):
    md = col.metadata
    assert md["embedding_model"] == "fake-model" and int(md["embedding_dimensions"]) == DIMS
    assert md["hnsw:space"] == "cosine"


def test_T1_STORE_rerun_is_idempotent(col):
    e = FakeEmbed()
    s1 = sync_paper_chunks(col, "1111.00001", paper(), e)
    assert (s1.added, s1.updated, s1.deleted) == (3, 0, 0) and col.count() == 3
    calls = e.calls
    s2 = sync_paper_chunks(col, "1111.00001", paper(), e)
    assert (s2.added, s2.updated, s2.deleted, s2.unchanged) == (0, 0, 0, 3)
    assert e.calls == calls and s2.embedded_tokens == 0, "unchanged content must not be re-embedded"
    assert col.count() == 3


def test_T1_STORE_changed_chunk_reembedded_only(col):
    e = FakeEmbed()
    sync_paper_chunks(col, "1111.00001", paper(), e)
    before = e.texts
    s = sync_paper_chunks(col, "1111.00001", paper(texts=("alpha text", "BETA CHANGED", "gamma text")), e)
    assert (s.added, s.updated, s.unchanged) == (0, 1, 2) and e.texts - before == 1
    got = col.get(ids=["1111.00001v1::S1::001"], include=["documents"])
    assert got["documents"] == ["BETA CHANGED"]


def test_T1_STORE_version_change_deletes_stale(col):
    e = FakeEmbed()
    sync_paper_chunks(col, "1111.00001", paper(v=1), e)
    sync_paper_chunks(col, "2222.00002", paper("2222.00002", 1), e)
    s = sync_paper_chunks(col, "1111.00001", paper(v=2, texts=("alpha text", "beta text")), e)
    assert s.added == 2 and s.deleted == 3
    ids = set(col.get(include=[])["ids"])
    assert not any(i.startswith("1111.00001v1") for i in ids)
    assert {"1111.00001v2::S1::000", "1111.00001v2::S1::001"} <= ids
    assert sum(i.startswith("2222.00002v1") for i in ids) == 3, "other paper untouched"


def test_T1_STORE_paper_becoming_metadata_only_removes_chunks(col):
    e = FakeEmbed()
    sync_paper_chunks(col, "1111.00001", paper(), e)
    s = sync_paper_chunks(col, "1111.00001", [], e)
    assert s.deleted == 3 and col.count() == 0


def test_T1_STORE_duplicate_chunk_ids_rejected(col):
    c = paper()
    with pytest.raises(ValueError):
        sync_paper_chunks(col, "1111.00001", c + [c[0]], FakeEmbed())
    assert col.count() == 0


def test_T1_STORE_prune(col):
    e = FakeEmbed()
    sync_paper_chunks(col, "1111.00001", paper(), e)
    sync_paper_chunks(col, "2222.00002", paper("2222.00002"), e)
    assert prune_papers(col, {"2222.00002"}) == 3
    assert {m["arxiv_id"] for m in col.get(include=["metadatas"])["metadatas"]} == {"2222.00002"}


def test_T1_STORE_vectors_persist_with_dims(tmp_path):
    c = open_collection(open_client(tmp_path / "c"), "tcol", "fake-model", DIMS)
    sync_paper_chunks(c, "1111.00001", paper(), FakeEmbed())
    c2 = open_collection(open_client(tmp_path / "c"), "tcol", "fake-model", DIMS)
    embs = c2.get(include=["embeddings"])["embeddings"]
    assert len(embs) == 3 and all(len(v) == DIMS for v in embs)


@pytest.mark.parametrize("model,dims", [("other-model", DIMS), ("fake-model", DIMS * 2)])
def test_T1_STORE_mismatch_refused(tmp_path, model, dims):
    cl = open_client(tmp_path / "c")
    c = open_collection(cl, "tcol", "fake-model", DIMS)
    sync_paper_chunks(c, "1111.00001", paper(), FakeEmbed())
    with pytest.raises(EmbeddingMismatch):
        open_collection(open_client(tmp_path / "c"), "tcol", model, dims)
    assert open_client(tmp_path / "c").get_collection("tcol").count() == 3, "refusal must not alter the index"


def test_T1_STORE_rebuild_recreates(tmp_path):
    cl = open_client(tmp_path / "c")
    c = open_collection(cl, "tcol", "fake-model", DIMS)
    sync_paper_chunks(c, "1111.00001", paper(), FakeEmbed())
    c2 = open_collection(cl, "tcol", "new-model", 16, rebuild=True)
    assert c2.count() == 0 and c2.metadata["embedding_model"] == "new-model"


# ------------------------------------------------------------------ SQLite

def rec(aid, authors, cats, primary, repo=None, v=1):
    return {"arxiv_id": aid, "version": v, "doc_id": f"{aid}v{v}", "title": f"T {aid}", "abstract": "a",
            "authors": authors, "published": "2023-05-01", "updated": "2023-06-01", "primary_category": primary,
            "categories": cats, "comment": "", "journal_ref": "", "doi": "", "abs_url": "x", "html_url": "y",
            "fulltext": True, "word_count": 10, "n_sections": 1, "n_chunks": 2, "topic": "rag", "repo": repo}


def test_T1_SQL_counts_and_dedup(tmp_path):
    db = tmp_path / "p.db"
    recs = [
        rec("1111.00001", ["Ann", "Bob", "Ann"], ["cs.CL", "cs.AI", "cs.CL"], "cs.CL",
            {"owner": "o", "name": "n", "url": "https://github.com/o/n", "found_in": "comment"}),
        rec("2222.00002", ["Bob", "Cy"], ["cs.AI"], "cs.AI"),
    ]
    counts = write_sqlite(db, recs)
    assert counts == {"papers": 2, "authors": 3, "paper_authors": 4, "paper_categories": 3, "repos": 1}
    con = sqlite3.connect(db)
    # independent truth: Bob authored 2 papers; Ann 1 (listed twice)
    q = "SELECT a.name, COUNT(*) FROM paper_authors pa JOIN authors a USING(author_id) GROUP BY a.name ORDER BY a.name"
    assert con.execute(q).fetchall() == [("Ann", 1), ("Bob", 2), ("Cy", 1)]
    assert con.execute("SELECT category, is_primary FROM paper_categories WHERE arxiv_id='1111.00001' ORDER BY category").fetchall() == [("cs.AI", 0), ("cs.CL", 1)]
    assert con.execute("SELECT year FROM papers").fetchall() == [(2023,), (2023,)]
    assert con.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_T1_SQL_rewrite_is_idempotent(tmp_path):
    db = tmp_path / "p.db"
    recs = [rec("1111.00001", ["Ann"], ["cs.CL"], "cs.CL")]
    assert write_sqlite(db, recs) == write_sqlite(db, recs)


def test_T1_SQL_failed_write_keeps_previous_db(tmp_path):
    db = tmp_path / "p.db"
    write_sqlite(db, [rec("1111.00001", ["Ann"], ["cs.CL"], "cs.CL")])
    dup = [rec("2222.00002", ["Ann"], ["cs.CL"], "cs.CL")] * 2  # duplicate primary key
    with pytest.raises(sqlite3.IntegrityError):
        write_sqlite(db, dup)
    con = sqlite3.connect(db)
    assert con.execute("SELECT arxiv_id FROM papers").fetchall() == [("1111.00001",)]


# ------------------------------------------------------------------ repos

@pytest.mark.parametrize("owner,name,exp", [
    ("org", "repo.git", ("org", "repo")),
    ("org", "repo.git.", ("org", "repo")),
    ("org", "repo).", ("org", "repo")),
    ("org", "repo,", ("org", "repo")),
    ("features", "copilot", None),
    ("orgs", "x", None),
])
def test_T1_REPO_normalize(owner, name, exp):
    assert _normalize(owner, name) == exp


def meta(comment="", abstract=""):
    return PaperMeta(arxiv_id="1111.00001", version=1, title="t", abstract=abstract, authors=[], published="2023-01-01",
                     updated="2023-01-01", primary_category="cs.CL", categories=["cs.CL"], comment=comment)


def html(intro="", front=""):
    return (f'<html><body><article class="ltx_document"><h1 class="ltx_title_document">T</h1>'
            f'<div class="ltx_authors">{front}</div><div class="ltx_abstract"><p>abs</p></div>'
            f'<section class="ltx_section" id="S1"><h2>Intro</h2>{intro}</section></article></body></html>')


def test_T1_REPO_priority_comment_over_abstract_over_html():
    m = meta(comment="Code: https://github.com/c/one.", abstract="see github.com/a/two")
    r = find_repo(m, html(front='<a href="https://github.com/h/three">x</a>'))
    assert (r["owner"], r["name"], r["found_in"], r["url"]) == ("c", "one", "comment", "https://github.com/c/one")
    assert find_repo(meta(abstract="at github.com/a/two."), None)["found_in"] == "abstract"
    assert find_repo(meta(), html(front='<a href="https://github.com/h/three">x</a>'))["found_in"] == "html_front_matter"


def test_T1_REPO_intro_requires_release_wording():
    third_party = '<p>We build on LangChain <a href="https://github.com/langchain-ai/langchain">link</a>.</p>'
    assert find_repo(meta(), html(intro=third_party)) is None
    own = '<p>Our code is available at <a href="https://github.com/me/proj">this repo</a>.</p>'
    assert find_repo(meta(), html(intro=own))["name"] == "proj"


def test_T1_REPO_github_io_not_a_repo():
    assert find_repo(meta(comment="Project page: https://selfrag.github.io/"), None) is None


# ------------------------------------------------------------------ arXiv Atom

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
 <entry><id>http://arxiv.org/abs/2310.11511v1</id><updated>2023-10-17T17:00:00Z</updated>
  <published>2023-10-17T17:00:00Z</published><title>Self-RAG:  Learning
  to Retrieve</title><summary> Abs text </summary>
  <author><name>Akari Asai</name></author><author><name>Zeqiu Wu</name></author>
  <arxiv:comment>30 pages</arxiv:comment>
  <arxiv:primary_category term="cs.CL"/><category term="cs.CL"/><category term="cs.AI"/></entry>
 <entry><id>http://arxiv.org/api/errors#incorrect_id_format_for_9999</id><title>Error</title></entry>
</feed>"""


def test_T1_ATOM_parse_and_skip_error_entry():
    out = parse_atom(ATOM)
    assert len(out) == 1
    m = out[0]
    assert (m.arxiv_id, m.version, m.title, m.published) == ("2310.11511", 1, "Self-RAG: Learning to Retrieve", "2023-10-17")
    assert m.authors == ["Akari Asai", "Zeqiu Wu"] and m.categories == ["cs.CL", "cs.AI"]


@pytest.mark.parametrize("raw,exp", [("2310.11511v3", ("2310.11511", 3)), ("2310.11511", ("2310.11511", None)),
                                     ("cs/0101001v2", ("cs/0101001", 2))])
def test_T1_ATOM_split_version(raw, exp):
    assert split_version(raw) == exp
