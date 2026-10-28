"""Phase 8 (UI polish): the answer Markdown parser under Node, static-page safety checks, and the event
fields the source viewer relies on (paper title/section on citations, the full web snippet, P2-N3/P5-N4).
Offline: scripted LLM responses and a fake MCP server, no provider calls."""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest
from langchain_core.messages import AIMessage

from app.config import ROOT
from app.graph import run_request
from app.mcp_tavily import SNIPPET_CHARS, parse_search_result, shared_search_tool
from tests.phase2_fakes import Script, drain, make_ctx, plan, tool_call

STATIC = ROOT / "app" / "static"
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def ev(events, name):
    return [p for e, p in events if e == name]


def node():
    n = shutil.which("node")
    if not n:
        pytest.skip("node not installed")
    return n


def test_markdown_parser_blocks_inline_injection_unicode_and_partial_input():
    r = subprocess.run([node(), str(ROOT / "tests/unit/js/md_check.mjs"), str(STATIC / "md.js")],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr + r.stdout
    assert '"failures":0' in r.stdout


@pytest.mark.parametrize("name", ["app.js", "md.js"])
def test_static_js_parses_and_never_builds_html_from_strings(name):
    assert subprocess.run([node(), "--check", str(STATIC / name)]).returncode == 0
    src = (STATIC / name).read_text()
    for bad in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert bad not in src, (name, bad)


def test_page_loads_parser_before_client_and_has_no_external_or_framework_assets():
    html = (STATIC / "index.html").read_text()
    scripts = re.findall(r'<script[^>]*src="([^"]+)"', html)
    assert scripts == ["/static/md.js", "/static/app.js"]
    assert not re.search(r'(src|href)="(https?:)?//', html)
    assert "single-turn" in html and "no conversation memory" in html
    assert not (ROOT / "node_modules").exists() and not (STATIC / "node_modules").exists()


def test_fault_token_is_cleared_from_the_address_bar_after_reading():
    src = (STATIC / "app.js").read_text()
    i = src.index("const FAULT = (")
    block = src[i:src.index("})();", i)]
    assert "history.replaceState" in block and "fault_token" in block


def test_client_has_a_stall_watchdog_and_settles_the_status_on_http_errors():
    src = (STATIC / "app.js").read_text()
    assert "AbortController" in src and "STALL_MS" in src
    # the HTTP-error path goes through fail(), which overwrites "Planning…" in place (P7-N2)
    assert re.search(r"if \(!r\.ok\)[\s\S]{0,600}throw new Error", src)
    assert re.search(r"const fail = \(msg\) => \{[\s\S]{0,200}status\.textContent = `Error: \$\{msg\}`", src)


async def test_paper_citation_carries_title_and_section_for_the_viewer():
    s = Script(planner=plan(("research", "explain alpha", [])),
               worker=[AIMessage(content="", tool_calls=[tool_call("corpus_search", {"query": "alpha"})]),
                       AIMessage(content="Alpha retrieves when uncertain [C1].")],
               synth_chunks=["Alpha retrieves ", "when uncertain [C1]."])
    ctx = make_ctx("How does alpha retrieve?", s)
    await run_request(ctx)
    (c,) = ev(drain(ctx), "citation")
    assert c["cite_id"] == "C1" and c["title"] and c["section"]
    assert c["label"] == f"{c['title']} — {c['section']}"
    assert c["title"] != c["chunk_id"]


async def test_web_search_event_and_registry_keep_the_full_snippet():
    long = "Lemon Agent reaches 91.69% on the benchmark. " * 40  # ~1,800 chars, cut to SNIPPET_CHARS
    results = parse_search_result([{"type": "text", "text": json.dumps({"query": "q", "results": [
        {"url": "https://example.org/lemon", "title": "Lemon", "content": long, "score": 0.9}]})}])

    class MCP:
        enabled = True

        async def search(self, query, max_results, time_range=None):
            return results

    ctx = make_ctx("q", Script())
    ctx.tavily = MCP()
    await shared_search_tool(ctx, "research", "test").coroutine(query="lemon agent")
    (ws,) = ev(drain(ctx), "web_search")
    snip = ws["results"][0]["snippet"]
    assert len(snip) == SNIPPET_CHARS == len(ctx.registry.by_cite["W1"].text)
    assert snip == ctx.registry.by_cite["W1"].text  # the viewer shows exactly what the model saw
