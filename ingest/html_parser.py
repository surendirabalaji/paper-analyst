"""Parse arXiv LaTeXML HTML into ordered section records.

Structure verified on 2210.03629, 2310.11511, 2401.15884 (Phase 0) and 2606.24535 (Phase 1):
<article class="ltx_document"> holds <h1 class="ltx_title_document">, <div class="ltx_abstract">,
and nested <section class="ltx_section|ltx_subsection|ltx_subsubsection|ltx_appendix"> elements
whose id attributes (S3, S3.SS2, A1.SS1) are stable anchors. Small run-in paragraphs
(ltx_paragraph, ids like S3.SS2.SSS1.Px1) are folded into their parent unit.

Text extraction joins inline elements with no separator, so small-caps headings split across
spans ("Re<span>asoning</span>") come out whole; separators are added only at block boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, NavigableString, Tag

UNIT_CLASSES = {"ltx_section", "ltx_subsection", "ltx_subsubsection", "ltx_appendix"}
SKIP_TITLES = re.compile(r"^(acknowledg|author contributions?\b)", re.IGNORECASE)
DROP_SELECTORS = [
    "nav",
    "header",
    "footer",
    "script",
    "style",
    "img",
    "button",
    ".ltx_bibliography",
    ".ltx_page_footer",
    ".ltx_page_header",
    ".ltx_authors",
    ".ltx_dates",
    ".ltx_note",  # footnotes and author notes
    ".ltx_tag_equation",
    ".ltx_tag_equationgroup",
    ".ltx_tag_listingline",  # algorithm line numbers
    ".ltx_listing_data",  # "⬇" download link for code listings
    ".ltx_acknowledgements",  # \\begin{acks} blocks folded into a conclusion
    ".ltx_ERROR",
    ".package-alerts",
]
BLOCK_TAGS = {
    "p", "div", "section", "figure", "figcaption", "table", "tr", "li", "ul", "ol",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "br", "dl", "dt", "dd",
}
CELL_TAGS = {"td", "th"}
MAX_SVG_LABEL_WORDS = 12  # inline token boxes ("Retrieve") are kept; diagrams are dropped
MAX_MATH_CHARS = 400
_PGF_RE = re.compile(r"\\(lxSVG|pgf|lx@)")
_WS_RE = re.compile(r"[ \t ]+")
_NL_RE = re.compile(r"\s*\n\s*")


@dataclass
class Section:
    section_id: str  # LaTeXML anchor ("S3.SS2") or "abstract"
    title: str  # heading without its number ("Self-Rag Training")
    number: str  # "3.2", "A.1" or ""
    level: int  # 0 abstract, 1 section/appendix, 2 subsection, 3 subsubsection
    text: str
    path: str = ""  # breadcrumb incl. ancestors: "3 Self-Rag › 3.2 Self-Rag Training"


@dataclass
class ParsedPaper:
    title: str
    sections: list[Section]

    @property
    def word_count(self) -> int:
        return sum(len(s.text.split()) for s in self.sections)


def _render(node: Tag | NavigableString, out: list[str]) -> None:
    if isinstance(node, NavigableString):
        if type(node) is NavigableString:  # skip comments, CDATA, doctype
            out.append(str(node))
        return
    if node.name == "svg":
        # LaTeXML draws TikZ (inline token boxes and whole diagrams) as SVG with text in
        # foreignObject spans. Keep short inline labels; drop diagram label soup.
        labels = [" ".join(c.get_text("").split()) for c in node.select(".ltx_foreignobject_content")]
        label = " ".join(x for x in labels if x)
        if label and len(label.split()) <= MAX_SVG_LABEL_WORDS:
            out.append(f" {label} ")
        return
    if node.name == "math":
        alt = (node.get("alttext") or "").strip()
        if _PGF_RE.search(alt) or len(alt) > MAX_MATH_CHARS:
            # alttext holds raw TikZ/pgf source when a drawn token sits inside math;
            # keep only the drawn labels.
            for svg in node.find_all("svg"):
                _render(svg, out)
            return
        out.append(f" {alt} ")
        return
    block = node.name in BLOCK_TAGS
    if block:
        out.append("\n")
    for child in node.children:
        _render(child, out)
    if node.name in CELL_TAGS:
        out.append(" | ")
    elif block:
        out.append("\n")


def clean_text(node: Tag) -> str:
    out: list[str] = []
    _render(node, out)
    text = _WS_RE.sub(" ", "".join(out))
    text = _NL_RE.sub("\n", text)
    text = re.sub(r"(\s*\|\s*)+\n", "\n", text)  # trailing cell separators
    return text.strip()


def _heading(unit: Tag) -> tuple[str, str]:
    h = unit.find(re.compile(r"^h[1-6]$"), recursive=False) or unit.find(class_=re.compile(r"^ltx_title"))
    if h is None:
        return "", ""
    tag = h.find(class_=re.compile(r"^ltx_tag"))
    number = clean_text(tag).strip().rstrip(".") if tag else ""
    if tag:
        tag.extract()
    title = " ".join(clean_text(h).split())
    h.extract()
    # LaTeXML prefixes appendix tags with "Appendix A"; keep the letter/number only.
    number = re.sub(r"^Appendix\s+", "", number)
    return title, number


def _level(unit: Tag) -> int:
    classes = set(unit.get("class", []))
    if classes & {"ltx_section", "ltx_appendix"}:
        return 1
    if "ltx_subsection" in classes:
        return 2
    return 3


def _is_unit(el: Tag) -> bool:
    return el.name == "section" and bool(UNIT_CLASSES & set(el.get("class", []))) and bool(el.get("id"))


def _drop_runin_acknowledgments(article: Tag) -> None:
    """Drop acknowledgments written as a run-in paragraph inside another section."""
    for px in article.select("section.ltx_paragraph"):
        h = px.find(re.compile(r"^h[1-6]$"))
        if h is not None and SKIP_TITLES.match(" ".join(h.get_text("").split())):
            px.decompose()
    # A paragraph whose whole text is "Acknowledgement(s)" (a bold pseudo-heading): drop it and
    # the paragraphs after it in the same section.
    for para in article.select(".ltx_para"):
        if para.parent is None:  # already removed with an earlier heading
            continue
        if re.fullmatch(r"(?i)acknowledge?ments?[.:]?", " ".join(para.get_text("").split())):
            for sib in list(para.find_next_siblings(class_="ltx_para")):
                sib.decompose()
            para.decompose()


def parse_html(html: str) -> ParsedPaper:
    soup = BeautifulSoup(html, "lxml")
    article = soup.find("article", class_="ltx_document")
    if article is None:
        raise ValueError("no LaTeXML <article class='ltx_document'> found")

    title_el = article.find(class_="ltx_title_document")
    title = " ".join(clean_text(title_el).split()) if title_el else ""

    for sel in DROP_SELECTORS:
        for el in article.select(sel):
            el.decompose()

    _drop_runin_acknowledgments(article)

    sections: list[Section] = []
    abstract = article.find(class_="ltx_abstract")
    if abstract is not None:
        for h in abstract.find_all(class_="ltx_title_abstract"):
            h.decompose()
        text = clean_text(abstract)
        if text:
            sections.append(Section("abstract", "Abstract", "", 0, text, "Abstract"))
        abstract.decompose()

    # Collect units in document order, then detach nested units so each unit keeps only
    # its own direct prose. Headings are read before detaching.
    units = [el for el in article.find_all("section") if _is_unit(el)]
    meta = []
    labels: dict[int, str] = {}
    for u in units:
        title_u, number = _heading(u)
        label = f"{number} {title_u}".strip() or u.get("id")
        labels[id(u)] = label
        parent = u.find_parent(_is_unit)
        path = f"{labels[id(parent)]} › {label}" if parent is not None and id(parent) in labels else label
        labels[id(u)] = path
        meta.append((u.get("id"), title_u, number, _level(u), path))
    for u in reversed(units):  # innermost first so parents lose fully-extracted children
        u.extract()
    for u, (sid, title_u, number, level, path) in zip(units, meta):
        if SKIP_TITLES.match(title_u.strip(" .")):
            continue
        text = clean_text(u)
        if text:
            sections.append(Section(sid, title_u or sid, number, level, text, path))
    return ParsedPaper(title=title, sections=sections)
