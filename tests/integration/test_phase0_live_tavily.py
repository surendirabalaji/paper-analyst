"""T0-LIVE-TAV: Tavily Remote MCP auth is proven by a non-error search payload, not by non-empty text.

Opt-in (RUN_LIVE=1); consumes one Tavily search credit. Never prints the key.
Regression for defect P0-D1: an invalid key yields list_tools success and a
tool *result* containing {"error": ..., "status": 401}, which must not count as success.
"""
import asyncio
import json
import os

import pytest

pytestmark = pytest.mark.live


def _search(key: str) -> dict:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    async def go():
        c = MultiServerMCPClient(
            {"t": {"transport": "streamable_http", "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={key}"}}
        )
        tools = {t.name: t for t in await c.get_tools()}
        assert "tavily_search" in tools
        r = await tools["tavily_search"].ainvoke({"query": "arXiv LLM agents", "max_results": 2})
        blocks = r if isinstance(r, list) else [{"text": r}]
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
        assert key not in text
        return json.loads(text)

    return asyncio.run(go())


def test_invalid_key_is_reported_as_error_payload():
    payload = _search("tvly-CANARYinvalid0000000000")
    assert payload.get("status") == 401 or "error" in payload


def test_real_key_returns_results(root):
    from dotenv import load_dotenv

    load_dotenv(root / ".env")
    key = os.environ.get("TAVILY_API_KEY", "")
    if not key or "REPLACE_ME" in key:
        pytest.fail("TAVILY_API_KEY not configured (BLOCKED, not a pass)")
    payload = _search(key)
    assert "error" not in payload
    assert len(payload.get("results", [])) >= 1
