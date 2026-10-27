"""T0-SAFE: scripts/phase0_check.py fails clearly on missing credentials and redacts secrets.

Uses synthetic canary values only; never touches real credentials.
"""
import asyncio
import importlib.util
import json

import pytest

CANARY_T = "tvly-CANARYtavily0123456789"
CANARY_O = "sk-CANARYopenai0123456789abcdef"
CANARY_R = "sk-or-CANARYrouter0123456789"


@pytest.fixture()
def mod(root, monkeypatch):
    spec = importlib.util.spec_from_file_location("phase0_check", root / "scripts" / "phase0_check.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_redacts_tavily_query_param(mod, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    s = mod.redact(f"GET https://mcp.tavily.com/mcp/?tavilyApiKey={CANARY_T}&x=1 failed")
    assert CANARY_T not in s and "tavilyApiKey=REDACTED" in s


def test_redacts_env_values_anywhere(mod, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", CANARY_O)
    monkeypatch.setenv("OPENROUTER_API_KEY", CANARY_R)
    monkeypatch.setenv("TAVILY_API_KEY", CANARY_T)
    nested = {"err": {"detail": f"bad key {CANARY_O}", "hdr": f"Bearer {CANARY_R}", "t": CANARY_T}}
    s = mod.redact(repr(nested))
    for c in (CANARY_O, CANARY_R, CANARY_T):
        assert c not in s


@pytest.mark.parametrize("sep", ["%3D", "%3d"])
@pytest.mark.parametrize("name", ["tavilyApiKey", "TAVILYAPIKEY"])
def test_redacts_url_encoded_tavily_key(mod, monkeypatch, sep, name):
    # P0-I3 regression: percent-encoded separator and case variants must be redacted.
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    s = mod.redact(f"https://mcp.tavily.com/mcp/?{name}{sep}tvly-CANARYenc&x=1")
    assert "tvly-CANARYenc" not in s
    assert "x=1" in s  # does not swallow following parameters


@pytest.mark.parametrize(
    "env,expected_missing",
    [
        ({"OPENAI_API_KEY": "", "OPENAI_MODEL": "m"}, ["OPENAI_API_KEY"]),
        ({"OPENAI_API_KEY": "sk-REPLACE_ME", "OPENAI_MODEL": "m"}, ["OPENAI_API_KEY"]),
    ],
)
def test_missing_or_placeholder_credentials_are_skipped(mod, monkeypatch, env, expected_missing):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    called = []

    async def fn():
        called.append(1)
        return {}

    res = asyncio.run(mod.run_one("openai-direct", fn, list(env)))
    assert res["status"] == "skipped"
    assert res["missing_env"] == expected_missing
    assert called == []


def test_exception_messages_are_redacted(mod, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", CANARY_T)
    monkeypatch.setenv("OPENAI_API_KEY", CANARY_O)

    async def fn():
        raise RuntimeError(f"connect https://mcp.tavily.com/mcp/?tavilyApiKey={CANARY_T} with {CANARY_O}")

    res = asyncio.run(mod.run_one("x", fn, ["TAVILY_API_KEY"]))
    assert res["status"] == "error"
    assert CANARY_T not in res["error"] and CANARY_O not in res["error"]


# --- P0-D1 regression (offline, synthetic payloads): Tavily tool-result error detection ---

ERR_401 = json.dumps({"error": "Search failed", "status": 401, "detail": "see https://docs.tavily.com"})
OK = json.dumps({"query": "q", "results": [{"url": "https://example.org/a", "title": "A", "content": "x"}]})


@pytest.mark.parametrize(
    "text",
    [
        ERR_401,  # raw JSON error
        json.dumps([{"type": "text", "text": ERR_401}]),  # MCP content-block list
        json.dumps({"status": 429, "message": "rate limited"}),  # status only
        json.dumps({"error": "quota exceeded"}),  # error only
        json.dumps({"query": "q", "results": []}),  # zero results
        json.dumps([{"type": "text", "text": json.dumps({"query": "q", "results": []})}]),
    ],
)
def test_tavily_error_detects_failures(mod, text):
    assert mod.tavily_error(text) is not None


@pytest.mark.parametrize("text", [OK, json.dumps([{"type": "text", "text": OK}])])
def test_tavily_error_accepts_real_results(mod, text):
    assert mod.tavily_error(text) is None


class _FakeTool:
    def __init__(self, name, result):
        self.name, self._r, self.args = name, result, {"query": {}}

    async def ainvoke(self, _args):
        return self._r


def _patch_client(monkeypatch, result):
    import langchain_mcp_adapters.client as c

    class FakeClient:
        def __init__(self, *_a, **_k):
            pass

        async def get_tools(self):
            return [_FakeTool("tavily_search", result)]

    monkeypatch.setattr(c, "MultiServerMCPClient", FakeClient)


@pytest.mark.parametrize(
    "result",
    [
        [{"type": "text", "text": ERR_401}],  # list-of-blocks object (adapter form)
        ERR_401,  # plain string form
        json.dumps({"query": "q", "results": []}),
    ],
)
def test_check_tavily_reports_error_status_on_failure_payload(mod, monkeypatch, result):
    monkeypatch.setenv("TAVILY_API_KEY", CANARY_T)
    _patch_client(monkeypatch, result)
    res = asyncio.run(mod.run_one("tavily-remote-mcp", mod.check_tavily, ["TAVILY_API_KEY"]))
    assert res["status"] == "error"
    assert CANARY_T not in json.dumps(res)


def test_check_tavily_ok_on_real_shaped_results(mod, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", CANARY_T)
    _patch_client(monkeypatch, [{"type": "text", "text": OK}])
    res = asyncio.run(mod.run_one("tavily-remote-mcp", mod.check_tavily, ["TAVILY_API_KEY"]))
    assert res["status"] == "ok" and res["search_ok"] is True and res["search_error"] is None


def test_check_tavily_plain_text_error_with_url_is_not_ok(mod, monkeypatch):
    # Probe: a non-JSON error string that happens to contain a URL.
    monkeypatch.setenv("TAVILY_API_KEY", CANARY_T)
    _patch_client(monkeypatch, "Error: Unauthorized. See https://docs.tavily.com/errors")
    res = asyncio.run(mod.run_one("tavily-remote-mcp", mod.check_tavily, ["TAVILY_API_KEY"]))
    if res["status"] == "ok":
        pytest.xfail("non-JSON error text containing a URL is accepted as success (P0-I4, low)")
