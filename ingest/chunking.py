"""Section-bounded token chunking with stable chunk IDs.

Chunks never cross a section boundary: each Section is split on its own with
RecursiveCharacterTextSplitter measured in cl100k_base tokens (the text-embedding-3-small
tokenizer). Each stored chunk is prefixed with "{paper title} — {section breadcrumb}" so short
sections stay retrievable, and the prefix is part of what gets embedded.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingest.html_parser import ParsedPaper

ENCODING = "cl100k_base"
CHUNK_TOKENS = 500
OVERLAP_TOKENS = 75
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

_enc = tiktoken.get_encoding(ENCODING)


def n_tokens(text: str) -> int:
    return len(_enc.encode(text, disallowed_special=()))


def make_splitter(chunk_tokens: int = CHUNK_TOKENS, overlap_tokens: int = OVERLAP_TOKENS):
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=ENCODING,
        chunk_size=chunk_tokens,
        chunk_overlap=overlap_tokens,
        separators=SEPARATORS,
        disallowed_special=(),
    )


@dataclass
class Chunk:
    chunk_id: str  # "{doc_id}::{section_id}::{n:03d}"
    text: str  # prefixed text: what is embedded and stored
    metadata: dict

    @property
    def sha256(self) -> str:
        return self.metadata["content_sha256"]


def chunk_paper(
    paper: ParsedPaper,
    *,
    arxiv_id: str,
    doc_id: str,
    title: str,
    published: str,
    html_url: str,
    splitter: RecursiveCharacterTextSplitter | None = None,
) -> list[Chunk]:
    splitter = splitter or make_splitter()
    chunks: list[Chunk] = []
    for order, sec in enumerate(paper.sections):
        pieces = [p.strip() for p in splitter.split_text(sec.text) if p.strip()]
        for n, piece in enumerate(pieces):
            text = f"{title} — {sec.path}\n{piece}"
            sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            anchor = "" if sec.section_id == "abstract" else f"#{sec.section_id}"
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::{sec.section_id}::{n:03d}",
                    text=text,
                    metadata={
                        "arxiv_id": arxiv_id,
                        "doc_id": doc_id,
                        "title": title,
                        "section_id": sec.section_id,
                        "section_title": sec.title,
                        "section_path": sec.path,
                        "section_number": sec.number,
                        "section_order": order,
                        "chunk_index": n,
                        "published": published,
                        "html_anchor_url": f"{html_url}{anchor}",
                        "n_tokens": n_tokens(text),
                        "content_sha256": sha,
                    },
                )
            )
    return chunks
