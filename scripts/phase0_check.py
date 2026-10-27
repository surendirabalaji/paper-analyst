"""Phase 0 setup check: prove every external service works before app code exists.

Checks (each independent; one failing does not stop the others):
  1. OpenAI direct: model listed, streaming call with a tool bound, usage returned; embeddings dims.
  2. Gemini via OpenRouter: same streaming + tool-call + usage check through the OpenRouter gateway.
  3. Tavily Remote MCP: Streamable HTTP, key as tavilyApiKey query param; list tools; one tavily_search call.

Usage:  .venv/bin/python scripts/phase0_check.py [--json out.json]
Key values are never printed. The Tavily URL is redacted in all output.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

TAVILY_KEY_RE = re.compile(r"tavilyApiKey(=|%3D)[^&\s'\"]+", re.IGNORECASE)


def redact(text: str) -> str:
    text = TAVILY_KEY_RE.sub(r"tavilyApiKey\1REDACTED", str(text))
    for var in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "TAVILY_API_KEY"):
        val = os.environ.get(var)
        if val and len(val) > 8:
            text = text.replace(val, f"<{var}>")
    return text


@tool
def get_paper_year(arxiv_id: str) -> int:
    """Return the publication year of an arXiv paper given its ID, e.g. '2310.11511'."""
    return 2000 + int(arxiv_id[:2])


TOOL_PROMPT = "Use the get_paper_year tool to find the publication year of arXiv paper 2310.11511."
STREAM_PROMPT = "In one sentence, what is retrieval-augmented generation?"


async def check_chat(label: str, llm: ChatOpenAI) -> dict:
    out: dict = {"service": label, "model_requested": llm.model_name}
    # Streaming text call: TTFT, chunk count, usage.
    t0 = time.perf_counter()
    ttft = None
    n_chunks = 0
    text = ""
    usage = None
    model_returned = None
    async for chunk in llm.astream(STREAM_PROMPT):
        if chunk.content and ttft is None:
            ttft = time.perf_counter() - t0
        if chunk.content:
            n_chunks += 1
            text += chunk.content if isinstance(chunk.content, str) else str(chunk.content)
        if chunk.usage_metadata:
            usage = dict(chunk.usage_metadata)
        model_returned = chunk.response_metadata.get("model_name") or model_returned
    out.update(
        stream_ok=bool(text),
        ttft_s=round(ttft, 3) if ttft else None,
        total_s=round(time.perf_counter() - t0, 3),
        content_chunks=n_chunks,
        usage=usage,
        model_returned=model_returned,
        sample=text[:120],
    )
    # Streaming tool call: does the gateway/model emit a well-formed tool call?
    bound = llm.bind_tools([get_paper_year])
    gathered = None
    async for chunk in bound.astream(TOOL_PROMPT):
        gathered = chunk if gathered is None else gathered + chunk
    calls = gathered.tool_calls if gathered is not None else []
    out["tool_call_ok"] = bool(calls) and calls[0]["name"] == "get_paper_year"
    out["tool_call"] = calls[0] if calls else None
    return out


async def check_openai() -> dict:
    model = os.environ["OPENAI_MODEL"]
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        )
        r.raise_for_status()
        ids = {m["id"] for m in r.json()["data"]}
    llm = ChatOpenAI(model=model, stream_usage=True, timeout=60, max_retries=1)
    out = await check_chat("openai-direct", llm)
    out["model_listed_in_/v1/models"] = model in ids
    emb_model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    vec = await OpenAIEmbeddings(model=emb_model).aembed_query("retrieval augmented generation")
    out["embedding"] = {"model": emb_model, "dims": len(vec), "listed": emb_model in ids}
    return out


async def check_openrouter() -> dict:
    model = os.environ["GEMINI_MODEL"]
    base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get(f"{base}/models")
        r.raise_for_status()
        entry = next((m for m in r.json()["data"] if m["id"] == model), None)
    llm = ChatOpenAI(
        model=model,
        base_url=base,
        api_key=os.environ["OPENROUTER_API_KEY"],
        stream_usage=True,
        timeout=60,
        max_retries=1,
        extra_body={"usage": {"include": True}},
    )
    out = await check_chat("gemini-via-openrouter", llm)
    out["model_listed_in_catalog"] = entry is not None
    if entry:
        out["catalog_pricing_per_M"] = {
            "input": round(float(entry["pricing"]["prompt"]) * 1e6, 4),
            "output": round(float(entry["pricing"]["completion"]) * 1e6, 4),
        }
    return out


async def check_tavily() -> dict:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    url = f"https://mcp.tavily.com/mcp/?tavilyApiKey={os.environ['TAVILY_API_KEY']}"
    client = MultiServerMCPClient({"tavily": {"transport": "streamable_http", "url": url}})
    t0 = time.perf_counter()
    tools = await client.get_tools()
    out: dict = {
        "service": "tavily-remote-mcp",
        "transport": "streamable_http",
        "auth": "tavilyApiKey query parameter",
        "list_tools_s": round(time.perf_counter() - t0, 3),
        "tools": {t.name: sorted((t.args or {}).keys()) for t in tools},
    }
    search = next((t for t in tools if t.name == "tavily_search"), None)
    if search is None:
        out["search_ok"] = False
        return out
    t0 = time.perf_counter()
    res = await search.ainvoke({"query": "latest LLM agent framework releases", "max_results": 3})
    text = res if isinstance(res, str) else json.dumps(res, default=str)
    # Tavily reports auth/quota failures as a normal tool result (e.g. {"error": ..., "status": 401}),
    # not as an exception, so success requires parsed results and no error marker.
    err = tavily_error(text)
    urls = re.findall(r"https?://[^\s\"'\\,\]]+", text)
    out.update(
        search_ok=err is None and bool(urls),
        search_error=err,
        search_s=round(time.perf_counter() - t0, 3),
        result_chars=len(text),
        sample_urls=urls[:3],
    )
    if not out["search_ok"]:
        raise RuntimeError(f"tavily_search returned no usable results: {err or 'zero result URLs'}")
    return out


def tavily_error(text: str) -> str | None:
    """Return an error description if a Tavily tool result signals failure, else None."""
    candidates = [text]
    try:
        outer = json.loads(text)
        # MCP content blocks: [{"type": "text", "text": "<json>"}]
        if isinstance(outer, list):
            candidates = [b.get("text", "") for b in outer if isinstance(b, dict)]
        elif isinstance(outer, dict):
            candidates = [json.dumps(outer)]
    except (json.JSONDecodeError, TypeError):
        pass
    for c in candidates:
        try:
            obj = json.loads(c)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            status = obj.get("status")
            if obj.get("error") or (isinstance(status, int) and status >= 400):
                return f"status={status} error={str(obj.get('error'))[:120]}"
            if "results" in obj and not obj["results"]:
                return "zero results"
    return None


async def run_one(name, fn, required_env):
    missing = [v for v in required_env if not os.environ.get(v) or "REPLACE_ME" in os.environ[v]]
    if missing:
        return {"service": name, "status": "skipped", "missing_env": missing}
    try:
        res = await fn()
        res["status"] = "ok"
        return res
    except Exception as e:  # report, don't crash the other checks
        return {"service": name, "status": "error", "error": redact(f"{type(e).__name__}: {e}")[:600]}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write results to this path")
    args = ap.parse_args()
    load_dotenv()
    import logging

    for noisy in ("httpx", "httpcore", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    results = await asyncio.gather(
        run_one("openai-direct", check_openai, ["OPENAI_API_KEY", "OPENAI_MODEL"]),
        run_one("gemini-via-openrouter", check_openrouter, ["OPENROUTER_API_KEY", "GEMINI_MODEL"]),
        run_one("tavily-remote-mcp", check_tavily, ["TAVILY_API_KEY"]),
    )
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "results": results,
    }
    text = redact(json.dumps(report, indent=2, default=str))
    print(text)
    if args.json:
        with open(args.json, "w") as f:
            f.write(text + "\n")
    return 0 if all(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
