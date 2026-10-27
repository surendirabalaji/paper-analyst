"""Build the corpus: arXiv metadata + HTML -> sections -> chunks -> Chroma, plus SQLite metadata.

Usage:
  .venv/bin/python scripts/ingest.py --dry-run      # fetch + parse + chunk, report counts and cost; no embedding
  .venv/bin/python scripts/ingest.py                # incremental build (embeds only new/changed chunks)
  .venv/bin/python scripts/ingest.py --rebuild      # drop the collection and re-embed everything

Inputs:  data/corpus_seed.json (committed paper list)
Cache:   data/raw/arxiv_metadata.json, data/raw/<doc_id>.html, data/raw/<doc_id>.fetch.json (git-ignored)
Outputs: CHROMA_PERSIST_DIR (collection CHROMA_COLLECTION), SQLITE_PATH, data/corpus_manifest.json

Re-running with unchanged inputs embeds nothing and adds 0 chunks. Paper versions are pinned by
data/corpus_manifest.json once it exists; --update-versions moves to the latest arXiv versions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from ingest.arxiv_client import ArxivClient, PaperMeta  # noqa: E402
from ingest.chunking import CHUNK_TOKENS, ENCODING, OVERLAP_TOKENS, chunk_paper, make_splitter  # noqa: E402
from ingest.html_parser import parse_html  # noqa: E402
from ingest.repos import find_repo  # noqa: E402

SEED = ROOT / "data" / "corpus_seed.json"
MANIFEST = ROOT / "data" / "corpus_manifest.json"
RAW = ROOT / "data" / "raw"
META_CACHE = RAW / "arxiv_metadata.json"
MIN_WORDS = 3000
EMBED_PRICE_PER_M = 0.02  # USD per 1M tokens, text-embedding-3-small, openai.com model page, checked 2026-10-26
PRICE_SOURCE = "https://developers.openai.com/api/docs/models/text-embedding-3-small (checked 2026-10-26)"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(p: str | Path) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def load_metadata(client: ArxivClient, ids: list[str], refresh: bool) -> dict[str, PaperMeta]:
    cache: dict[str, dict] = {}
    if META_CACHE.exists() and not refresh:
        cache = json.loads(META_CACHE.read_text())
    missing = [i for i in ids if i not in cache]
    if missing:
        print(f"Fetching arXiv metadata for {len(missing)} papers")
        for k, m in client.metadata(missing).items():
            cache[k] = m.__dict__
        META_CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True))
    return {k: PaperMeta(**v) for k, v in cache.items() if k in ids}


def fetch_html(client: ArxivClient, doc_id: str, refetch: bool) -> dict:
    html_path, side = RAW / f"{doc_id}.html", RAW / f"{doc_id}.fetch.json"
    if side.exists() and not refetch:
        info = json.loads(side.read_text())
        if info["status"] != 200 or html_path.exists():
            info["html"] = html_path.read_text() if info["status"] == 200 else None
            return info
    status, url, text = client.html(doc_id)
    info = {"status": status, "url": url, "fetched_at": now()}
    if status == 200:
        html_path.write_text(text)
    side.write_text(json.dumps(info))
    return {**info, "html": text if status == 200 else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="no embedding, no index/DB writes")
    ap.add_argument("--rebuild", action="store_true", help="drop and recreate the Chroma collection")
    ap.add_argument("--refetch", action="store_true", help="ignore the HTML cache")
    ap.add_argument("--refresh-metadata", action="store_true", help="ignore the arXiv metadata cache")
    ap.add_argument("--update-versions", action="store_true", help="use latest arXiv versions, not manifest pins")
    ap.add_argument("--limit", type=int, default=0, help="only the first N seed papers (testing)")
    ap.add_argument("--max-embed-tokens", type=int, default=8_000_000, help="abort if more would be embedded")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    persist = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma_index")
    persist = str((ROOT / persist).resolve()) if not os.path.isabs(persist) else persist
    collection = os.environ.get("CHROMA_COLLECTION", "arxiv_agents")
    sqlite_path = os.environ.get("SQLITE_PATH", "./data/papers.db")
    sqlite_path = str((ROOT / sqlite_path).resolve()) if not os.path.isabs(sqlite_path) else sqlite_path

    seed = json.loads(SEED.read_text())
    unique: dict[str, dict] = {}
    for e in seed["papers"]:
        if e["arxiv_id"] in unique:
            print(f"WARNING: duplicate arxiv_id {e['arxiv_id']} in {rel(SEED)}; keeping the first entry")
            continue
        unique[e["arxiv_id"]] = e
    entries = list(unique.values())[: args.limit or None]
    ids = [e["arxiv_id"] for e in entries]
    topic = {e["arxiv_id"]: e["topic"] for e in entries}
    pins: dict[str, int] = {}
    if MANIFEST.exists() and not args.update_versions:
        pins = {p["arxiv_id"]: p["version"] for p in json.loads(MANIFEST.read_text())["papers"]}

    RAW.mkdir(parents=True, exist_ok=True)
    client = ArxivClient()
    t0 = time.monotonic()
    metas = load_metadata(client, ids, args.refresh_metadata)
    unknown = [i for i in ids if i not in metas]
    if unknown:
        print(f"WARNING: arXiv returned no metadata for {unknown}")

    splitter = make_splitter()
    records, plans = [], []
    for n, aid in enumerate(ids, 1):
        if aid not in metas:
            continue
        m = metas[aid]
        version = pins.get(aid, m.version)
        doc_id = f"{aid}v{version}"
        html_url = f"https://arxiv.org/html/{doc_id}"
        fetch = fetch_html(client, doc_id, args.refetch)
        rec = {
            "arxiv_id": aid, "version": version, "doc_id": doc_id, "title": m.title, "abstract": m.abstract,
            "authors": m.authors, "published": m.published, "updated": m.updated,
            "primary_category": m.primary_category, "categories": m.categories, "comment": m.comment,
            "journal_ref": m.journal_ref, "doi": m.doi, "license": m.license,
            "abs_url": f"https://arxiv.org/abs/{doc_id}", "html_url": html_url, "topic": topic[aid],
            "fetched_at": fetch.get("fetched_at"), "html_status": fetch["status"],
            "fulltext": False, "status": "metadata_only", "reason": "", "word_count": 0, "n_sections": 0, "n_chunks": 0,
        }
        chunks = []
        if fetch["status"] != 200:
            rec["reason"] = f"no HTML (HTTP {fetch['status']})"
        else:
            try:
                parsed = parse_html(fetch["html"])
                rec["word_count"], rec["n_sections"] = parsed.word_count, len(parsed.sections)
                if parsed.word_count < MIN_WORDS:
                    rec["reason"] = f"parsed text {parsed.word_count} words < {MIN_WORDS}"
                else:
                    chunks = chunk_paper(parsed, arxiv_id=aid, doc_id=doc_id, title=m.title,
                                         published=m.published, html_url=html_url, splitter=splitter)
                    rec.update(fulltext=True, status="indexed", n_chunks=len(chunks))
            except Exception as exc:  # malformed page: keep metadata, log, continue
                rec["status"], rec["reason"] = "parse_failed", f"{type(exc).__name__}: {exc}"
        rec["repo"] = find_repo(m, fetch.get("html"))
        records.append(rec)
        plans.append((rec, chunks))
        print(f"[{n:3}/{len(ids)}] {doc_id:14} {rec['status']:13} words={rec['word_count']:6} "
              f"chunks={rec['n_chunks']:4} repo={'y' if rec['repo'] else '-'} {rec['reason']}")
    client.close()

    all_chunks = [c for _, cs in plans for c in cs]
    total_tokens = sum(c.metadata["n_tokens"] for c in all_chunks)
    print(f"\nPapers: {len(records)} | indexed: {sum(r['fulltext'] for r in records)} | "
          f"chunks: {len(all_chunks)} | chunk tokens: {total_tokens:,} | arXiv requests: {client.n_requests}")
    print(f"Full re-embed cost estimate: ${total_tokens / 1e6 * EMBED_PRICE_PER_M:.4f} at ${EMBED_PRICE_PER_M}/1M tokens")
    if args.dry_run:
        print("Dry run: nothing embedded or written.")
        return 0

    from langchain_openai import OpenAIEmbeddings

    from ingest.store import open_client, open_collection, prune_papers, sync_paper_chunks, write_sqlite

    emb = OpenAIEmbeddings(model=model)
    client_db = open_client(persist)
    dims = None
    if args.rebuild or collection not in {c.name for c in client_db.list_collections()}:
        # Measure the model's dimensions before creating or dropping anything, so a model change
        # can never leave a dropped or mislabelled collection behind. Existing collections use
        # their stored dimensions, so a no-op rerun makes no embedding calls.
        dims = len(emb.embed_documents(["dimension probe"])[0])
    col = open_collection(client_db, collection, model, dims, rebuild=args.rebuild)
    dims = int(col.metadata["embedding_dimensions"])

    def embed(texts: list[str]) -> list[list[float]]:
        vecs = emb.embed_documents(texts)
        bad = [len(v) for v in vecs if len(v) != dims]
        if bad:
            raise RuntimeError(f"embedding dims {bad[0]} != {dims}")
        return vecs

    before = col.count()
    have = set(col.get(include=[])["ids"]) if before else set()
    planned = sum(c.metadata["n_tokens"] for c in all_chunks if c.chunk_id not in have)
    if planned > args.max_embed_tokens:
        print(f"ABORT: would embed ~{planned:,} new-chunk tokens > --max-embed-tokens {args.max_embed_tokens:,}")
        return 2

    totals = {"added": 0, "updated": 0, "deleted": 0, "unchanged": 0, "embedded_tokens": 0}
    pruned = 0 if args.limit else prune_papers(col, {r["arxiv_id"] for r in records if r["fulltext"]})
    for rec, chunks in plans:
        st = sync_paper_chunks(col, rec["arxiv_id"], chunks, embed)
        for k in totals:
            totals[k] += getattr(st, k)
    after = col.count()
    # A --limit run is a partial test build: never overwrite the full metadata DB with it.
    db_counts = write_sqlite(sqlite_path, records) if not args.limit else {"skipped": "--limit run"}

    run = {
        "finished_at": now(), "seconds": round(time.monotonic() - t0, 1), "chunks_before": before,
        "chunks_after": after, **totals, "pruned_other_papers": pruned,
        "embedding_cost_usd": round(totals["embedded_tokens"] / 1e6 * EMBED_PRICE_PER_M, 5),
    }
    manifest = {
        "generated_at": now(),
        "seed": rel(SEED),
        "source": "arXiv LaTeXML HTML (https://arxiv.org/html/<id>v<n>) + arXiv Atom API metadata",
        "selection": {"min_words_for_fulltext": MIN_WORDS, "criteria": seed["selection_criteria"]},
        "chunking": {"splitter": "RecursiveCharacterTextSplitter.from_tiktoken_encoder", "encoding": ENCODING,
                     "chunk_tokens": CHUNK_TOKENS, "overlap_tokens": OVERLAP_TOKENS,
                     "boundary": "never crosses a section; prefix '{title} — {section path}' is embedded",
                     "chunk_id": "{doc_id}::{section_id}::{n:03d}"},
        "embedding": {"model": model, "dimensions": dims, "price_usd_per_1m_tokens": EMBED_PRICE_PER_M,
                      "price_source": PRICE_SOURCE},
        "vector_store": {"engine": "chroma", "persist_dir": rel(persist), "collection": collection,
                         "distance": "cosine", "count": after},
        "sqlite": {"path": rel(sqlite_path), "tables": db_counts},
        "totals": {
            "papers": len(records),
            "fulltext_indexed": sum(r["fulltext"] for r in records),
            "metadata_only": sum(not r["fulltext"] for r in records),
            "chunks": len(all_chunks),
            "chunk_tokens": total_tokens,
            "words_indexed": sum(r["word_count"] for r in records if r["fulltext"]),
            "papers_with_repo": sum(bool(r["repo"]) for r in records),
            "full_embed_cost_usd": round(total_tokens / 1e6 * EMBED_PRICE_PER_M, 5),
        },
        "last_run": run,
        "papers": [
            {k: r[k] for k in ("arxiv_id", "version", "doc_id", "title", "published", "updated", "primary_category",
                               "topic", "license", "html_url", "fetched_at", "html_status", "status", "fulltext",
                               "reason", "word_count", "n_sections", "n_chunks", "repo")}
            for r in records
        ],
    }
    if not args.limit:
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"last_run": run, "sqlite": db_counts}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
