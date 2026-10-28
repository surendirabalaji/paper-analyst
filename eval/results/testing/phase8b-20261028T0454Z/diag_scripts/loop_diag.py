"""Diagnose 'Event loop is closed': real OpenAI embedding calls across two event loops (cost ~$0.000001)."""
import asyncio, sys, time
sys.path.insert(0, ".")
from app.config import load_settings
from app.rag import Retriever

s = load_settings()
r = Retriever(s)
mode = sys.argv[1]

async def loop1():
    t = asyncio.create_task(r.search("how does self-rag decide when to retrieve", 5))
    if mode == "await":
        await t
    elif mode == "abandon":
        await asyncio.sleep(0.05)  # task still in flight when the loop ends (kept-but-unconsumed)
    elif mode == "sleep_done":
        await asyncio.sleep(3)     # finished but never awaited

async def loop2():
    try:
        out = await r.search("reflexion verbal feedback", 3)
        return f"ok {len(out)}"
    except Exception as e:
        return f"ERR {type(e).__name__}: {e}"

asyncio.run(loop1())
print(mode, "->", asyncio.run(loop2()))
