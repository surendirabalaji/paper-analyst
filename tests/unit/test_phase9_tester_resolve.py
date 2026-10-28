"""Phase 9 tester: held-out checks of the resolve_paper abstract fallback against the real corpus DB.

Expected values come from independent SQL over data/papers.db (computed in the test), not from the
implementation. The DB is opened read-only by the code under test; nothing is written.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from app.agents.summary import PaperNotFound, resolve_paper

DB = Path(__file__).resolve().parents[2] / "data/papers.db"
pytestmark = pytest.mark.skipif(not DB.exists(), reason="corpus DB not built")


def abstract_hits(name: str, case: bool) -> list[str]:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    pat = re.compile(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", 0 if case else re.I)
    rows = [a for a, ab in con.execute("SELECT arxiv_id, abstract FROM papers") if pat.search(ab)]
    con.close()
    return rows


def title_hits(name: str) -> list[str]:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = [r[0] for r in con.execute("SELECT arxiv_id FROM papers WHERE title LIKE ?", (f"%{name}%",))]
    con.close()
    return rows


@pytest.mark.parametrize("name", ["MMOA-RAG", "HyDE"])
def test_method_only_in_abstract_resolves_to_the_unique_paper(name):
    assert not title_hits(name)
    truth = abstract_hits(name, case=len(name) <= 4)
    assert len(truth) == 1
    row = resolve_paper(DB, name)
    assert row["arxiv_id"] == truth[0] and row.get("resolved_by") == "abstract"


def test_baseline_name_absent_from_every_abstract_is_not_resolved():
    # R1-Searcher appears only in TeaRAG's body text; there is no R1-Searcher paper.
    assert not title_hits("R1-Searcher") and not abstract_hits("R1-Searcher", case=False)
    with pytest.raises(PaperNotFound):
        resolve_paper(DB, "R1-Searcher")


def test_phrase_in_many_abstracts_is_ambiguous_not_a_silent_pick():
    q = "retrieval-augmented generation"
    assert not title_hits(q) or len(title_hits(q)) > 1
    if title_hits(q):
        pytest.skip("phrase also in titles; title path decides")
    assert len(abstract_hits(q, case=False)) > 1
    with pytest.raises(PaperNotFound, match="several"):
        resolve_paper(DB, q)


def test_short_name_is_case_exact():
    assert abstract_hits("HyDE", case=True)
    with pytest.raises(PaperNotFound):
        resolve_paper(DB, "hyde")


@pytest.mark.parametrize("q", ["%%%", "a_b_c_d", ".*RAG.*", "') OR 1=1 --"])
def test_wildcards_and_regex_metacharacters_do_not_match_everything(q):
    with pytest.raises(PaperNotFound):
        resolve_paper(DB, q)


def test_title_match_still_takes_precedence_over_abstract():
    # Reflexion is a title; its abstract also names it. The title path must return the same paper
    # without the abstract marker.
    row = resolve_paper(DB, "Reflexion")
    assert row["arxiv_id"] in title_hits("Reflexion") and "resolved_by" not in row
