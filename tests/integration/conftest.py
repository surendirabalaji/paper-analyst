"""Live integration test support.

P8b-D1: anyio gives every test its own event loop, while the OpenAI SDK's async client closes itself from
`__del__` by scheduling `aclose()` on whatever loop is running then, and langchain-openai caches its default
async httpx client process-wide. A client created in one test and collected during a later one closes a
connection that belongs to a dead loop ("Event loop is closed"). This fixture drops those clients at the
end of each live test while the test's own loop is still running, so their close tasks run on it.
Production is unaffected: the server runs one event loop for its lifetime.
"""

import asyncio
import gc

import pytest


@pytest.fixture
async def close_openai_clients():
    yield
    from langchain_openai.chat_models import _client_utils as cu

    for name in ("_cached_async_httpx_client", "_cached_sync_httpx_client"):
        fn = getattr(cu, name, None)
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()
    gc.collect()
    for _ in range(3):  # let the scheduled aclose() tasks finish on this loop
        await asyncio.sleep(0.05)
