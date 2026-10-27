"""Find a paper's own GitHub repository link (links only; no snapshots in Phase 1).

Priority: arXiv comment field, then the abstract, then the HTML front matter (title/author
notes, abstract block) and the introduction. Introduction links only count when release
wording ("code", "available", "released", ...) appears close by, so third-party tools the paper
merely mentions are not recorded as its code.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from ingest.arxiv_client import GITHUB_RE, PaperMeta

NON_REPO_OWNERS = {"features", "orgs", "topics", "sponsors", "user-attachments", "about", "marketplace", "settings"}
RELEASE_WORDS = re.compile(r"\b(code|codes|available|release[sd]?|open[- ]?source[sd]?|repository|implementation|github)\b", re.I)
CONTEXT_CHARS = 200


def _normalize(owner: str, name: str) -> tuple[str, str] | None:
    name = re.sub(r"\.git$", "", name.rstrip(".,;:)]}'\""))
    owner = owner.rstrip(".,;:)]}'\"")
    if not owner or not name or owner.lower() in NON_REPO_OWNERS:
        return None
    return owner, name


def _first(text: str) -> tuple[str, str] | None:
    for m in GITHUB_RE.finditer(text or ""):
        hit = _normalize(m.group(1), m.group(2))
        if hit:
            return hit
    return None


def _html_candidates(html: str) -> tuple[tuple[str, str] | None, str]:
    soup = BeautifulSoup(html, "lxml")
    article = soup.find("article", class_="ltx_document")
    if article is None:
        return None, ""
    front = [article.find(class_="ltx_title_document"), article.find(class_="ltx_authors"), article.find(class_="ltx_abstract")]
    for el in front:
        if el is None:
            continue
        hrefs = " ".join(a.get("href", "") for a in el.find_all("a")) + " " + el.get_text(" ")
        hit = _first(hrefs)
        if hit:
            return hit, "html_front_matter"
    intro = article.find("section", id="S1")
    if intro is not None:
        for a in intro.find_all("a", href=GITHUB_RE):
            hit = _first(a["href"])
            if not hit:
                continue
            # Look for release wording near the link, including its footnote text.
            host = a.find_parent(class_="ltx_note") or a.parent
            context = host.get_text(" ")[-CONTEXT_CHARS:] if host else ""
            prev = a.find_previous(string=True)
            context += " " + (str(prev)[-CONTEXT_CHARS:] if prev else "")
            if RELEASE_WORDS.search(context):
                return hit, "html_front_matter"
    return None, ""


def find_repo(meta: PaperMeta, html: str | None) -> dict | None:
    for field_name, text in (("comment", meta.comment), ("abstract", meta.abstract)):
        hit = _first(text)
        if hit:
            return _row(hit, field_name)
    if html:
        hit, where = _html_candidates(html)
        if hit:
            return _row(hit, where)
    return None


def _row(hit: tuple[str, str], found_in: str) -> dict:
    owner, name = hit
    return {"owner": owner, "name": name, "url": f"https://github.com/{owner}/{name}", "found_in": found_in}
