"""Phase 1 unit tests: LaTeXML parser and section-bounded chunker (synthetic fixtures, no network).

IDs: T1-PARSE-*, T1-CHUNK-*.
"""
from __future__ import annotations

import hashlib

import pytest

from ingest.chunking import CHUNK_TOKENS, OVERLAP_TOKENS, chunk_paper, n_tokens
from ingest.html_parser import parse_html

PGF_ALT = r"s(\mathchoice{\hbox to26pt{\pgfpicture\lxSVG@begingroup@{x}}})" + "x" * 50
TOKEN_SVG = '<svg><foreignObject><span class="ltx_foreignobject_content">Retrieve</span></foreignObject></svg>'
DIAGRAM_SVG = (
    "<svg>"
    + "".join(f'<foreignObject><span class="ltx_foreignobject_content">label{i} word word</span></foreignObject>' for i in range(10))
    + "</svg>"
)

FIXTURE = f"""<!DOCTYPE html><html><head><title>t</title><script>var x=1;</script></head><body>
<nav class="ltx_page_navbar">Back to arXiv</nav>
<header>Report issue for preceding element</header>
<article class="ltx_document">
 <h1 class="ltx_title ltx_title_document">Self-R<span class="ltx_font_smallcaps">ag</span>: A Test Paper</h1>
 <div class="ltx_authors">Alice Author<span class="ltx_note">Equal contribution</span></div>
 <div class="ltx_abstract"><h6 class="ltx_title ltx_title_abstract">Abstract</h6>
   <p>We study retrieval. Ünïcode — “quotes” and 漢字 survive.</p></div>
 <section class="ltx_section" id="S1"><h2 class="ltx_title"><span class="ltx_tag ltx_tag_section">1 </span>Intro<span class="ltx_font_smallcaps">duction</span></h2>
   <p>Intro prose with math <math alttext="x^{{2}}+y"><mi>x</mi></math> inline.</p>
   <p>Footnote here<span class="ltx_note">FOOTNOTE-NOISE</span>.</p>
   <section class="ltx_subsection" id="S1.SS1"><h3 class="ltx_title"><span class="ltx_tag">1.1 </span>Tokens</h3>
     <p>The model emits {TOKEN_SVG} when needed.</p>
     <p>Drawn equation <math alttext="{PGF_ALT}">{TOKEN_SVG}</math> done.</p>
     <figure>{DIAGRAM_SVG}<figcaption>Figure caption text</figcaption></figure>
     <div class="ltx_para ltx_paragraph" id="S1.SS1.SSS0.Px1"><h4 class="ltx_title">Run-in.</h4><p>Run-in paragraph prose.</p></div>
     <table><tr><td>cellA</td><td>cellB</td></tr></table>
     <div class="ltx_listing"><div class="ltx_listing_data"><a href="data:x">⬇</a></div><div>code line</div></div>
   </section>
 </section>
 <section class="ltx_section" id="S2"><h2 class="ltx_title"><span class="ltx_tag">2 </span>Empty Parent</h2>
   <section class="ltx_subsection" id="S2.SS1"><h3 class="ltx_title"><span class="ltx_tag">2.1 </span>Child</h3><p>Child prose.</p></section>
 </section>
 <section class="ltx_section" id="Sx1"><h2 class="ltx_title">Acknowledgments</h2><p>ACK-NOISE thanks.</p></section>
 <section class="ltx_appendix" id="A1"><h2 class="ltx_title"><span class="ltx_tag">Appendix A </span>Extra</h2><p>Appendix prose.</p></section>
 <section class="ltx_bibliography" id="bib"><h2>References</h2><ul><li>BIB-NOISE 2020</li></ul></section>
</article>
<footer>FOOTER-NOISE</footer>
</body></html>"""


@pytest.fixture(scope="module")
def parsed():
    return parse_html(FIXTURE)


def _sec(parsed, sid):
    return next(s for s in parsed.sections if s.section_id == sid)


def test_T1_PARSE_sections_order_ids_numbers(parsed):
    ids = [s.section_id for s in parsed.sections]
    # S2 has no prose of its own -> omitted; acknowledgments dropped; bibliography is not a unit.
    assert ids == ["abstract", "S1", "S1.SS1", "S2.SS1", "A1"]
    assert _sec(parsed, "S1").number == "1" and _sec(parsed, "S1.SS1").number == "1.1"
    assert _sec(parsed, "A1").number == "A", "appendix prefix 'Appendix ' should be stripped"
    assert [s.level for s in parsed.sections] == [0, 1, 2, 2, 1]


def test_T1_PARSE_smallcaps_title_and_heading_whole(parsed):
    assert parsed.title == "Self-Rag: A Test Paper"
    assert _sec(parsed, "S1").title == "Introduction"


def test_T1_PARSE_breadcrumb_keeps_empty_parent_title(parsed):
    assert _sec(parsed, "S2.SS1").path == "2 Empty Parent › 2.1 Child"
    assert _sec(parsed, "S1.SS1").path == "1 Introduction › 1.1 Tokens"


def test_T1_PARSE_noise_removed(parsed):
    full = "\n".join(s.text for s in parsed.sections)
    for noise in ("FOOTNOTE-NOISE", "BIB-NOISE", "ACK-NOISE", "FOOTER-NOISE", "Back to arXiv", "Report issue",
                  "var x=1", "Equal contribution", "⬇", "lxSVG", "pgfpicture"):
        assert noise not in full, noise


def test_T1_PARSE_math_svg_and_tables(parsed):
    s1 = _sec(parsed, "S1").text
    assert "x^{2}+y" in s1, "math alttext should be kept"
    ss = _sec(parsed, "S1.SS1").text
    assert "emits Retrieve when" in " ".join(ss.split()), "short inline SVG label kept"
    assert "Drawn equation Retrieve done" in " ".join(ss.split()), "pgf math replaced by drawn label"
    assert "label0" not in ss, "large diagram label soup dropped"
    assert "Figure caption text" in ss
    assert "Run-in paragraph prose" in ss, "run-in paragraph folded into parent unit"
    assert "cellA | cellB" in ss
    assert "code line" in ss


def test_T1_PARSE_nested_units_not_duplicated(parsed):
    assert "Child prose" not in "\n".join(s.text for s in parsed.sections if s.section_id != "S2.SS1")
    assert "Tokens" not in _sec(parsed, "S1").text


def test_T1_PARSE_unicode_preserved(parsed):
    a = _sec(parsed, "abstract").text
    assert "Ünïcode — “quotes” and 漢字" in a


def test_T1_PARSE_missing_article_raises():
    # arXiv 404 body / "no HTML" page: no LaTeXML structure.
    with pytest.raises(ValueError):
        parse_html("<html><body><h1>No HTML for '2310.99999'</h1></body></html>")


def test_T1_PARSE_empty_article_gives_no_sections():
    p = parse_html('<html><body><article class="ltx_document"></article></body></html>')
    assert p.sections == [] and p.word_count == 0


def test_T1_PARSE_ack_variant_heading_dropped():
    html = ('<html><body><article class="ltx_document"><section class="ltx_section" id="Sx1">'
            '<h2 class="ltx_title">Acknowledgments and Disclosure of Funding</h2><p>We thank X.</p></section>'
            "</article></body></html>")
    assert parse_html(html).sections == []


def _doc(body):
    return f'<html><body><article class="ltx_document">{body}</article></body></html>'


def test_T1_PARSE_ack_runin_forms_dropped_content_kept():
    """P1-D3 (fixed, retest 20261027T0459Z): run-in acknowledgments inside a conclusion are dropped,
    the conclusion's own prose is kept, and a following section is untouched."""
    html = _doc(
        '<section class="ltx_section" id="S5"><h2 class="ltx_title">5 Conclusion</h2>'
        '<div class="ltx_para"><p>KEEP-CONCLUSION prose.</p></div>'
        '<div class="ltx_para"><p><span class="ltx_text ltx_font_bold">Acknowledgement</span></p></div>'
        '<div class="ltx_para"><p>ACK-NOISE-1 we thank.</p></div>'
        '<div class="ltx_para"><p>ACK-NOISE-2 funding.</p></div>'
        '<section class="ltx_paragraph" id="S5.SS0.SSS0.Px1"><h4 class="ltx_title">Acknowledgements.</h4>'
        '<p>ACK-NOISE-3 grant.</p></section>'
        '<div class="ltx_acknowledgements">ACK-NOISE-4 acks env.</div>'
        '</section>'
        '<section class="ltx_section" id="S6"><h2 class="ltx_title">6 Limitations</h2><p>KEEP-LIMITS prose.</p></section>'
        '<section class="ltx_section" id="Sx2"><h2 class="ltx_title">Author Contributions</h2><p>ACK-NOISE-5</p></section>'
    )
    p = parse_html(html)
    text = " ".join(s.text for s in p.sections)
    assert "KEEP-CONCLUSION" in text and "KEEP-LIMITS" in text
    assert "ACK-NOISE" not in text
    assert {s.section_id for s in p.sections} == {"S5", "S6"}


def test_T1_PARSE_ack_word_in_ordinary_prose_kept():
    """Negative control: prose mentioning acknowledgment, and a bold term that is not the bare word, survive."""
    html = _doc(
        '<section class="ltx_section" id="S2"><h2 class="ltx_title">2 Method</h2>'
        '<div class="ltx_para"><p><b>Acknowledgement tokens</b> signal receipt.</p></div>'
        '<div class="ltx_para"><p>KEEP-AFTER we acknowledge prior work.</p></div></section>'
    )
    text = " ".join(s.text for s in parse_html(html).sections)
    assert "Acknowledgement tokens" in text and "KEEP-AFTER" in text


# ------------------------------------------------------------------ chunking

def _chunks(parsed, **kw):
    return chunk_paper(parsed, arxiv_id="9999.00001", doc_id="9999.00001v2", title="Self-Rag: A Test Paper",
                       published="2024-01-01", html_url="https://arxiv.org/html/9999.00001v2", **kw)


def test_T1_CHUNK_ids_metadata_hash_anchor(parsed):
    chunks = _chunks(parsed)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    for c in chunks:
        md = c.metadata
        assert c.chunk_id == f"{md['doc_id']}::{md['section_id']}::{md['chunk_index']:03d}"
        for k in ("doc_id", "section_id", "section_title", "chunk_index", "html_anchor_url", "content_sha256"):
            assert k in md and md[k] not in (None, "") or k == "chunk_index"
        assert md["content_sha256"] == hashlib.sha256(c.text.encode()).hexdigest()
        assert md["n_tokens"] == n_tokens(c.text)
        exp = "https://arxiv.org/html/9999.00001v2" + ("" if md["section_id"] == "abstract" else f"#{md['section_id']}")
        assert md["html_anchor_url"] == exp
        assert c.text.startswith(f"Self-Rag: A Test Paper — {md['section_path']}\n")


def test_T1_CHUNK_deterministic(parsed):
    a = [(c.chunk_id, c.sha256) for c in _chunks(parsed)]
    b = [(c.chunk_id, c.sha256) for c in _chunks(parse_html(FIXTURE))]
    assert a == b


def _long_paper():
    paras = "".join(f"<p>Sentence number {i} about retrieval augmented agents and planning. " * 6 + "</p>" for i in range(60))
    html = (f'<html><body><article class="ltx_document"><h1 class="ltx_title_document">Long</h1>'
            f'<section class="ltx_section" id="S1"><h2 class="ltx_title">One</h2>{paras}</section>'
            f'<section class="ltx_section" id="S2"><h2 class="ltx_title">Two</h2><p>SECTION-TWO-ONLY short text.</p></section>'
            "</article></body></html>")
    return parse_html(html)


def test_T1_CHUNK_size_overlap_and_section_boundary():
    p = _long_paper()
    chunks = chunk_paper(p, arxiv_id="1", doc_id="1v1", title="Long", published="2024-01-01", html_url="u")
    s1 = [c for c in chunks if c.metadata["section_id"] == "S1"]
    s2 = [c for c in chunks if c.metadata["section_id"] == "S2"]
    assert len(s1) >= 4 and len(s2) == 1
    assert [c.metadata["chunk_index"] for c in s1] == list(range(len(s1)))
    prefix = n_tokens("Long — One\n")
    for c in s1:
        body = c.text.split("\n", 1)[1]
        assert n_tokens(body) <= CHUNK_TOKENS
        assert c.metadata["n_tokens"] <= CHUNK_TOKENS + prefix + 1
        assert "SECTION-TWO-ONLY" not in c.text, "chunk crossed a section boundary"
    # consecutive chunks overlap (some text from the end of chunk i begins chunk i+1)
    overlaps = 0
    for a, b in zip(s1, s1[1:]):
        ta, tb = a.text.split("\n", 1)[1], b.text.split("\n", 1)[1]
        if tb[:40] in ta:
            overlaps += 1
    assert overlaps >= len(s1) - 2, "expected ~75-token overlap between consecutive chunks"
    assert OVERLAP_TOKENS == 75 and CHUNK_TOKENS == 500
