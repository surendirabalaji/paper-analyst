"""Phase 1 integration: scripts/ingest.py main() end to end on temp outputs.

Real cached arXiv HTML/metadata (data/raw, read only), FAKE embeddings, temp Chroma/SQLite/manifest/seed.
No network: the arXiv client is replaced by a stub that fails on any request, OPENAI_API_KEY is a fake.

IDs: T1-CLI-*.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys

import chromadb
import pytest



class NoNetworkClient:
    n_requests = 0

    def __init__(self, *a, **k):
        pass

    def metadata(self, ids, *a, **k):
        raise AssertionError(f"unexpected arXiv metadata request {ids}")

    def html(self, doc_id):
        raise AssertionError(f"unexpected arXiv HTML request {doc_id}")

    def close(self):
        pass


class FakeEmbeddings:
    calls = 0
    texts = 0
    dims = 1536

    def __init__(self, model=None, **k):
        self.model = model

    def embed_documents(self, texts):
        FakeEmbeddings.calls += 1
        FakeEmbeddings.texts += len(texts)
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            out.append([h[i % 32] / 255 + 0.001 for i in range(FakeEmbeddings.dims)])
        return out


@pytest.fixture()
def env(root, tmp_path, monkeypatch):
    if not (root / "data" / "raw" / "arxiv_metadata.json").exists():
        pytest.skip("raw arXiv cache not present")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-not-a-key")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHROMA_COLLECTION", "test_col")
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "papers.db"))
    import langchain_openai

    monkeypatch.setattr(langchain_openai, "OpenAIEmbeddings", FakeEmbeddings)
    FakeEmbeddings.calls = FakeEmbeddings.texts = 0
    FakeEmbeddings.dims = 1536
    return tmp_path


def load(root, tmp_path, monkeypatch, ids, fetch_override=None):
    spec = importlib.util.spec_from_file_location(f"ingest_cli_{len(ids)}_{id(tmp_path)}", root / "scripts" / "ingest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps({"selection_criteria": ["test"], "papers": [{"arxiv_id": i, "topic": "t"} for i in ids]}))
    monkeypatch.setattr(mod, "SEED", seed)
    monkeypatch.setattr(mod, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(mod, "ArxivClient", NoNetworkClient)
    if fetch_override:
        orig = mod.fetch_html

        def fh(client, doc_id, refetch):
            r = fetch_override(doc_id)
            return r if r is not None else orig(client, doc_id, refetch)

        monkeypatch.setattr(mod, "fetch_html", fh)
    return mod


def run(mod, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["ingest.py", *argv])
    return mod.main()


def col(tmp):
    return chromadb.PersistentClient(path=str(tmp / "chroma"), settings=chromadb.Settings(anonymized_telemetry=False)).get_collection("test_col")


def manifest(tmp):
    return json.loads((tmp / "manifest.json").read_text())


SMALL = ["2309.15217", "2307.07924", "2508.10152"]
META_ONLY = "2506.16988"


def test_T1_CLI_build_then_rerun_adds_zero(root, env, monkeypatch):
    mod = load(root, env, monkeypatch, SMALL)
    assert run(mod, monkeypatch) == 0
    m1 = manifest(env)
    n = m1["totals"]["chunks"]
    assert n > 0 and col(env).count() == n == m1["last_run"]["added"]
    calls = FakeEmbeddings.calls
    mod = load(root, env, monkeypatch, SMALL)
    assert run(mod, monkeypatch) == 0
    lr = manifest(env)["last_run"]
    assert (lr["added"], lr["updated"], lr["deleted"], lr["unchanged"], lr["embedded_tokens"]) == (0, 0, 0, n, 0)
    assert FakeEmbeddings.calls == calls and col(env).count() == n
    db = sqlite3.connect(env / "papers.db")
    assert db.execute("SELECT COUNT(*), SUM(n_chunks) FROM papers").fetchone() == (3, n)


def test_T1_CLI_metadata_only_failed_and_malformed_inputs(root, env, monkeypatch):
    def fake(doc_id):
        if doc_id.startswith("2307.07924"):
            return {"status": 404, "url": "u", "fetched_at": "t", "html": None}
        if doc_id.startswith("2508.10152"):
            return {"status": 200, "url": "u", "fetched_at": "t", "html": "<html><body>conversion failed</body></html>"}
        return None

    mod = load(root, env, monkeypatch, SMALL + [META_ONLY], fake)
    assert run(mod, monkeypatch) == 0
    m = manifest(env)
    st = {p["arxiv_id"]: (p["status"], p["fulltext"], p["n_chunks"], p["reason"]) for p in m["papers"]}
    assert st["2307.07924"][:3] == ("metadata_only", False, 0) and "404" in st["2307.07924"][3]
    assert st["2508.10152"][:3] == ("parse_failed", False, 0)
    assert st[META_ONLY][:3] == ("metadata_only", False, 0) and "< 3000" in st[META_ONLY][3]
    assert st["2309.15217"][:2] == ("indexed", True)
    assert m["totals"]["fulltext_indexed"] == 1 and m["totals"]["metadata_only"] == 3
    assert {md["arxiv_id"] for md in col(env).get(include=["metadatas"])["metadatas"]} == {"2309.15217"}
    db = sqlite3.connect(env / "papers.db")
    assert db.execute("SELECT COUNT(*), SUM(fulltext) FROM papers").fetchone() == (4, 1)


def test_T1_CLI_removed_from_seed_is_pruned_and_paper_losing_html_is_cleaned(root, env, monkeypatch):
    mod = load(root, env, monkeypatch, SMALL)
    run(mod, monkeypatch)
    # drop one paper from the seed, make another lose its HTML
    def fake(doc_id):
        if doc_id.startswith("2307.07924"):
            return {"status": 404, "url": "u", "fetched_at": "t", "html": None}
        return None

    mod = load(root, env, monkeypatch, ["2309.15217", "2307.07924"], fake)
    run(mod, monkeypatch)
    lr = manifest(env)["last_run"]
    ids = {md["arxiv_id"] for md in col(env).get(include=["metadatas"])["metadatas"]}
    assert ids == {"2309.15217"}
    assert lr["pruned_other_papers"] + lr["deleted"] > 0


def test_T1_CLI_version_change_replaces_old_version_chunks(root, env, monkeypatch):
    mod = load(root, env, monkeypatch, ["2309.15217"])
    run(mod, monkeypatch)
    m = manifest(env)
    old = m["papers"][0]["doc_id"]
    n = m["totals"]["chunks"]
    newv = m["papers"][0]["version"] + 1
    m["papers"][0]["version"] = newv  # pin a (simulated) newer version
    (env / "manifest.json").write_text(json.dumps(m))
    html = (root / "data" / "raw" / f"{old}.html").read_text()
    mod = load(root, env, monkeypatch, ["2309.15217"],
               lambda d: {"status": 200, "url": "u", "fetched_at": "t", "html": html} if d.endswith(f"v{newv}") else None)
    run(mod, monkeypatch)
    got = col(env).get(include=["metadatas"])
    assert col(env).count() == n
    assert all(i.startswith(f"2309.15217v{newv}::") for i in got["ids"])
    assert all(md["html_anchor_url"].startswith(f"https://arxiv.org/html/2309.15217v{newv}") for md in got["metadatas"])
    lr = manifest(env)["last_run"]
    assert lr["added"] == n and lr["deleted"] == n
    db = sqlite3.connect(env / "papers.db")
    assert db.execute("SELECT version, doc_id FROM papers").fetchone() == (newv, f"2309.15217v{newv}")


def test_T1_CLI_embedding_model_mismatch_refused_without_writes(root, env, monkeypatch):
    from ingest.store import EmbeddingMismatch

    mod = load(root, env, monkeypatch, ["2309.15217"])
    run(mod, monkeypatch)
    db_hash = hashlib.sha256((env / "papers.db").read_bytes()).hexdigest()
    n = col(env).count()
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    mod = load(root, env, monkeypatch, ["2309.15217"])
    with pytest.raises(EmbeddingMismatch, match="--rebuild"):
        run(mod, monkeypatch)
    assert col(env).count() == n
    assert hashlib.sha256((env / "papers.db").read_bytes()).hexdigest() == db_hash


def test_T1_CLI_query_index_refuses_model_mismatch(root, env, monkeypatch):
    import subprocess

    mod = load(root, env, monkeypatch, ["2309.15217"])
    run(mod, monkeypatch)
    import os

    e = {**os.environ, "OPENAI_EMBEDDING_MODEL": "text-embedding-3-large", "OPENAI_API_KEY": "sk-fake-not-a-key"}
    r = subprocess.run([sys.executable, str(root / "scripts" / "query_index.py"), "q"], env=e, capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "rebuild" in r.stdout


def test_T1_CLI_limit_does_not_desync_sqlite_from_index(root, env, monkeypatch):
    """P1-D1 (fixed, retest 20261027T0459Z): a --limit run must not rewrite papers.db or the manifest."""
    mod = load(root, env, monkeypatch, SMALL)
    run(mod, monkeypatch)
    db_hash = hashlib.sha256((env / "papers.db").read_bytes()).hexdigest()
    man_hash = hashlib.sha256((env / "manifest.json").read_bytes()).hexdigest()
    mod = load(root, env, monkeypatch, SMALL)
    assert run(mod, monkeypatch, "--limit", "1") == 0
    db = sqlite3.connect(env / "papers.db")
    in_index = {md["arxiv_id"] for md in col(env).get(include=["metadatas"])["metadatas"]}
    in_db = {r[0] for r in db.execute("SELECT arxiv_id FROM papers WHERE fulltext=1")}
    assert in_db == in_index == set(SMALL)
    assert hashlib.sha256((env / "papers.db").read_bytes()).hexdigest() == db_hash
    assert hashlib.sha256((env / "manifest.json").read_bytes()).hexdigest() == man_hash


def test_T1_CLI_duplicate_seed_id_handled(root, env, monkeypatch, capsys):
    """P1-D2 (fixed): duplicate seed id is dropped with a warning; exactly one paper, no duplicate chunks."""
    mod = load(root, env, monkeypatch, ["2309.15217", "2309.15217"])
    assert run(mod, monkeypatch) == 0
    assert "duplicate arxiv_id 2309.15217" in capsys.readouterr().out
    db = sqlite3.connect(env / "papers.db")
    assert db.execute("SELECT COUNT(*) FROM papers").fetchone() == (1,)
    ids = col(env).get(include=[])["ids"]
    assert len(ids) == len(set(ids)) == manifest(env)["totals"]["chunks"]
    assert manifest(env)["totals"]["papers"] == 1


def test_T1_CLI_rebuild_with_new_model_dims_probed_before_drop(root, env, monkeypatch):
    """P1-D4 (fixed): --rebuild for a model with different dims records the probed dims and stores matching vectors."""
    mod = load(root, env, monkeypatch, ["2309.15217"])
    run(mod, monkeypatch)
    n = col(env).count()
    assert n > 0 and col(env).metadata["embedding_dimensions"] == 1536
    FakeEmbeddings.dims = 3072
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    mod = load(root, env, monkeypatch, ["2309.15217"])
    assert run(mod, monkeypatch, "--rebuild") == 0
    c = col(env)
    assert c.metadata["embedding_model"] == "text-embedding-3-large"
    assert c.metadata["embedding_dimensions"] == 3072 == manifest(env)["embedding"]["dimensions"]
    assert c.count() == n
    vecs = c.get(include=["embeddings"], limit=5)["embeddings"]
    assert all(len(v) == 3072 for v in vecs)
    # unchanged rerun on the rebuilt collection: no probe, no embeds
    calls = FakeEmbeddings.calls
    mod = load(root, env, monkeypatch, ["2309.15217"])
    assert run(mod, monkeypatch) == 0
    assert FakeEmbeddings.calls == calls


def test_T1_CLI_rebuild_probe_failure_leaves_index_intact(root, env, monkeypatch):
    """P1-D4 (fixed): if the dims probe fails (e.g. provider error), --rebuild must not drop the collection."""
    mod = load(root, env, monkeypatch, ["2309.15217"])
    run(mod, monkeypatch)
    n = col(env).count()

    def boom(self, texts):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(FakeEmbeddings, "embed_documents", boom)
    mod = load(root, env, monkeypatch, ["2309.15217"])
    with pytest.raises(RuntimeError, match="provider unavailable"):
        run(mod, monkeypatch, "--rebuild")
    assert col(env).count() == n and col(env).metadata["embedding_dimensions"] == 1536


def test_T1_CLI_noop_rerun_makes_zero_embedding_calls_including_probe(root, env, monkeypatch):
    mod = load(root, env, monkeypatch, ["2309.15217"])
    run(mod, monkeypatch)
    assert FakeEmbeddings.calls >= 2  # probe + content batches on create
    calls = FakeEmbeddings.calls
    mod = load(root, env, monkeypatch, ["2309.15217"])
    run(mod, monkeypatch)
    assert FakeEmbeddings.calls == calls


def test_T1_STORE_create_without_dims_rejected(tmp_path):
    from ingest.store import open_client, open_collection

    client = open_client(str(tmp_path / "c"))
    with pytest.raises(ValueError, match="dimensions"):
        open_collection(client, "x", "text-embedding-3-small", None)
    assert "x" not in {c.name for c in client.list_collections()}


def test_T1_CLI_query_index_refuses_dims_mismatch(root, env, monkeypatch, capsys):
    """P1-D4 (fixed): same model name but the query embedding has other dims -> exit 2, no search."""
    mod = load(root, env, monkeypatch, ["2309.15217"])
    run(mod, monkeypatch)
    spec = importlib.util.spec_from_file_location("query_index_t", root / "scripts" / "query_index.py")
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)

    class FakeQ(FakeEmbeddings):
        def embed_query(self, t):
            return [0.1] * 3072

    monkeypatch.setattr(q, "OpenAIEmbeddings", FakeQ)
    monkeypatch.setattr(q, "ROOT", env.parent)  # persist dir env var is absolute; ROOT / abs -> abs
    monkeypatch.setattr(sys, "argv", ["query_index.py", "q"])
    assert q.main() == 2
    assert "rebuild" in capsys.readouterr().out
