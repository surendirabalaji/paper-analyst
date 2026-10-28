"""Real planner calls, each in its own event loop (like anyio tests). ~$0.002 per call."""
import asyncio, os, sys, traceback
sys.path.insert(0, ".")
from app.config import load_settings
from app.context import RequestContext
from app.events import Emitter
from app.llm.router import Router
from app.rag import Retriever
from app.graph import planner

s = load_settings()
QS = ["Who are the authors of the ReAct paper?", "How many papers in the corpus have a linked GitHub repository?",
      "How does Self-RAG decide when to retrieve?", "Compare how Reflexion and Self-Refine use feedback."]
n = int(sys.argv[1]); fails = 0
for i in range(n):
    q = QS[i % 4]
    async def one():
        ctx = RequestContext(request_id=f"diag{i}", query=q, k=s.top_k, settings=s, router=Router(s),
                             retriever=Retriever(s), emitter=Emitter("d"), tavily=None)
        out = await planner({"query": q}, {"configurable": {"ctx": ctx}})
        return [t["worker"] for t in out["tasks"]], (ctx.speculative or {}).get("status")
    try:
        print(i, q[:40], asyncio.run(one()))
    except Exception as e:
        fails += 1
        print(i, q[:40], "EXC", type(e).__name__, e)
        traceback.print_exc(limit=-12)
print("spec", s.speculative_retrieval, "fails", fails, "of", n)
