"""Tester Phase 2 (T2-JS): the browser's real SSE parser (app/static/app.js makeParser) under Node with
every single split point, 2,000 random multi-splits, 1-byte chunks, split multi-byte characters,
keep-alive comments and CRLF framing. Skips (not passes) when Node is unavailable."""
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_T2_JS1_parser_handles_arbitrary_network_splits():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    r = subprocess.run([node, str(ROOT / "tests/unit/js/sse_parser_check.mjs"), str(ROOT / "app/static/app.js")],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr + r.stdout
    assert '"failures":0' in r.stdout


def test_T2_JS2_app_js_syntax_and_no_innerhtml():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    assert subprocess.run([node, "--check", str(ROOT / "app/static/app.js")]).returncode == 0
    src = (ROOT / "app/static/app.js").read_text()
    assert "innerHTML" not in src and "insertAdjacentHTML" not in src and "eval(" not in src
