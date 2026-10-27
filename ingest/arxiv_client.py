"""Polite arXiv client: Atom API metadata and LaTeXML HTML pages.

arXiv's API terms (checked 2026-10-26, https://info.arxiv.org/help/api/tou.html): no more
than one request every three seconds, one connection at a time. Since 2026-10 arXiv also
answers bursts with HTTP 429 "Rate exceeded" without a Retry-After header, so every request
goes through one throttled connection with exponential backoff on 429/503.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import httpx

API_URL = "https://export.arxiv.org/api/query"
HTML_URL = "https://arxiv.org/html/{id}"
USER_AGENT = "csci599-paper-analyst/0.1 (academic project; mailto:surendir@usc.edu)"
MIN_INTERVAL_S = 3.5
BACKOFF_S = (20, 45, 90, 180)

_NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
_VERSION_RE = re.compile(r"^(?P<base>.+?)(v(?P<v>\d+))?$")
GITHUB_RE = re.compile(r"github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)")


class FetchError(RuntimeError):
    pass


@dataclass
class PaperMeta:
    arxiv_id: str  # base id, e.g. "2310.11511"
    version: int
    title: str
    abstract: str
    authors: list[str]
    published: str  # YYYY-MM-DD (v1)
    updated: str
    primary_category: str
    categories: list[str]
    comment: str = ""
    journal_ref: str = ""
    doi: str = ""
    license: str = ""
    links: list[str] = field(default_factory=list)

    @property
    def doc_id(self) -> str:
        return f"{self.arxiv_id}v{self.version}"


def split_version(raw_id: str) -> tuple[str, int | None]:
    m = _VERSION_RE.match(raw_id.strip())
    base = m.group("base")
    return base, int(m.group("v")) if m.group("v") else None


def _text(el: ET.Element | None) -> str:
    return " ".join(el.text.split()) if el is not None and el.text else ""


def parse_atom(xml_text: str) -> list[PaperMeta]:
    root = ET.fromstring(xml_text)
    out: list[PaperMeta] = []
    for e in root.findall("a:entry", _NS):
        raw = _text(e.find("a:id", _NS)).rsplit("/abs/", 1)[-1]
        base, version = split_version(raw)
        if not base or version is None:
            continue  # arXiv returns an error entry for unknown ids
        out.append(
            PaperMeta(
                arxiv_id=base,
                version=version,
                title=_text(e.find("a:title", _NS)),
                abstract=_text(e.find("a:summary", _NS)),
                authors=[_text(a.find("a:name", _NS)) for a in e.findall("a:author", _NS)],
                published=_text(e.find("a:published", _NS))[:10],
                updated=_text(e.find("a:updated", _NS))[:10],
                primary_category=e.find("x:primary_category", _NS).attrib.get("term", ""),
                categories=[c.attrib["term"] for c in e.findall("a:category", _NS)],
                comment=_text(e.find("x:comment", _NS)),
                journal_ref=_text(e.find("x:journal_ref", _NS)),
                doi=_text(e.find("x:doi", _NS)),
                license=(e.find("a:rights", _NS).text or "") if e.find("a:rights", _NS) is not None else "",
                links=[l.attrib.get("href", "") for l in e.findall("a:link", _NS)],
            )
        )
    return out


class ArxivClient:
    def __init__(self, min_interval_s: float = MIN_INTERVAL_S, backoff_s=BACKOFF_S, log=print):
        self._http = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=60,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=1),
        )
        self._min_interval = min_interval_s
        self._backoff = backoff_s
        self._last = 0.0
        self._log = log
        self.n_requests = 0

    def close(self) -> None:
        self._http.close()

    def _get(self, url: str, params: dict | None = None) -> httpx.Response:
        for attempt in range(len(self._backoff) + 1):
            wait = self._min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                r = self._http.get(url, params=params)
            except httpx.TransportError as exc:
                r, err = None, exc
            self._last = time.monotonic()
            self.n_requests += 1
            if r is not None and r.status_code not in (429, 500, 502, 503, 504):
                return r
            if attempt == len(self._backoff):
                break
            reason = r.status_code if r is not None else type(err).__name__
            self._log(f"  arXiv {reason} on {url}; backing off {self._backoff[attempt]}s")
            time.sleep(self._backoff[attempt])
        raise FetchError(f"gave up on {url} after {len(self._backoff) + 1} attempts")

    def metadata(self, ids: list[str], batch: int = 50) -> dict[str, PaperMeta]:
        """Latest-version metadata keyed by base arXiv id. Unknown ids are absent."""
        out: dict[str, PaperMeta] = {}
        for i in range(0, len(ids), batch):
            chunk = [split_version(x)[0] for x in ids[i : i + batch]]
            r = self._get(API_URL, {"id_list": ",".join(chunk), "max_results": len(chunk)})
            if r.status_code != 200 or not r.text.lstrip().startswith("<?xml"):
                raise FetchError(f"arXiv API returned {r.status_code}")
            for m in parse_atom(r.text):
                out[m.arxiv_id] = m
        return out

    def search(self, query: str, max_results: int = 25) -> list[PaperMeta]:
        r = self._get(API_URL, {"search_query": query, "max_results": max_results, "sortBy": "relevance"})
        if r.status_code != 200:
            raise FetchError(f"arXiv API returned {r.status_code}")
        return parse_atom(r.text)

    def html(self, doc_id: str) -> tuple[int, str, str]:
        """Fetch LaTeXML HTML for a versioned id. Returns (status, final_url, text)."""
        r = self._get(HTML_URL.format(id=doc_id))
        return r.status_code, str(r.url), r.text if r.status_code == 200 else ""
