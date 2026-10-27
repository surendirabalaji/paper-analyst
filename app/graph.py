"""LangGraph supervisor graph.

    START -> planner -> supervisor --route_next--> research | synthesize
                         ^              |
                         +--- worker ---+          synthesize -> END

The planner is one structured-output LLM call that produces a small task list. The supervisor is
deterministic: `route_next` reads task statuses, dependencies, retry counts and the request budget,
and every decision is emitted as a `transition` event with its reason. Workers always return to the
supervisor, and an exception or timeout inside a worker becomes a recorded `failed` result.
Phase 2 registers one worker (Research); Summary and Data are added in Phases 5 and 6.
"""

from __future__ import annotations

import asyncio
import operator
import time
from typing import Annotated, Any, Awaitable, Callable, TypedDict

import openai
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.agents.prompts import PLANNER_SYSTEM, SYNTH_NO_TASK_ROLE, SYNTH_ROLE
from app.agents.research import run_research
from app.context import RequestContext, ctx_of
from app.events import safe_error, utc_now
from app.llm.router import record_llm_result, text_of, traced_invoke
from app.rag import AUGMENT_PROMPT, cited_ids, format_evidence

MAX_TASKS = 4
MAX_ATTEMPTS = 2
TRANSIENT = {"timeout", "rate_limit", "provider_unavailable"}
DONE_LIKE = {"done", "partial"}

WorkerFn = Callable[[RequestContext, dict, str], Awaitable[dict]]
WORKERS: dict[str, tuple[WorkerFn, str]] = {
    "research": (
        run_research,
        "research: answers conceptual and cross-paper questions by vector search over the full text of the "
        "indexed papers, with citations.",
    ),
}


class AnalystState(TypedDict, total=False):
    request_id: str
    query: str
    k: int
    plan_summary: str
    plan_source: str  # "llm" or "fallback"
    tasks: list[dict]  # {id, worker, instruction, depends_on, status, attempts, error_kind}
    worker_results: dict[str, dict]
    next: str
    active_task: str | None
    step_count: int
    transitions: Annotated[list[dict], operator.add]
    errors: Annotated[list[dict], operator.add]
    final_answer: str
    status: str
    unresolved_citations: list[str]


# ---------------------------------------------------------------- planner


class PlannedTask(BaseModel):
    worker: str = Field(..., description="Name of an available worker")
    instruction: str = Field(..., description="What this worker should find or produce")
    depends_on: list[int] = Field(default_factory=list, description="0-based indices of earlier tasks it needs")


class Plan(BaseModel):
    plan_summary: str
    tasks: list[PlannedTask] = Field(default_factory=list)


def validate_plan(plan: Plan, query: str) -> tuple[list[dict], list[str]]:
    """Keep only known workers, at most MAX_TASKS, and dependencies on earlier kept tasks."""
    notes, tasks, index_map = [], [], {}
    for i, t in enumerate(plan.tasks):
        if t.worker not in WORKERS:
            notes.append(f"dropped task {i}: unknown worker {t.worker!r}")
            continue
        if len(tasks) >= MAX_TASKS:
            notes.append(f"dropped task {i}: task cap {MAX_TASKS}")
            continue
        tid = f"t{len(tasks) + 1}"
        index_map[i] = tid
        deps = [index_map[d] for d in t.depends_on if d in index_map and d < i]
        tasks.append(
            {
                "id": tid,
                "worker": t.worker,
                "instruction": (t.instruction or query).strip()[:600],
                "depends_on": deps,
                "status": "pending",
                "attempts": 0,
                "error_kind": None,
            }
        )
    return tasks, notes


def fallback_tasks(query: str) -> list[dict]:
    return [
        {"id": "t1", "worker": "research", "instruction": query, "depends_on": [], "status": "pending",
         "attempts": 0, "error_kind": None}
    ]


async def planner(state: AnalystState, config) -> dict:
    ctx = ctx_of(config)
    ctx.emitter.emit("transition", **{"from": "start", "to": "planner", "reason": "new request", "step": 0})
    decision = ctx.router.decide("planner", "planner", {"query_chars": len(state["query"])})
    system = PLANNER_SYSTEM.format(max_tasks=MAX_TASKS, workers="\n".join(f"- {d}" for _, d in WORKERS.values()))
    llm = ctx.router.client(decision.provider).with_structured_output(Plan, method="function_calling", include_raw=True)
    errors, source = [], "llm"
    try:
        out = await asyncio.wait_for(
            traced_invoke(ctx, decision, llm, [SystemMessage(system), HumanMessage(state["query"])]),
            timeout=ctx.settings.llm_timeout_s,
        )
        plan = out.get("parsed")
        if plan is None:
            raise ValueError(f"planner output did not parse: {out.get('parsing_error')}")
        tasks, notes = validate_plan(plan, state["query"])
        summary = plan.plan_summary.strip()[:300]
    except Exception as e:
        tasks, notes, source = fallback_tasks(state["query"]), [], "fallback"
        summary = "Planner unavailable; falling back to a single Research task."
        msg = safe_error(e)
        errors.append({"scope": "planner", "error_kind": "planner_error", "message": msg, "ts": utc_now()})
        ctx.emitter.emit("error", scope="planner", message_safe=msg, recoverable=True)
    ctx.emitter.emit(
        "plan",
        plan_summary=summary,
        source=source,
        notes=notes,
        tasks=[{k: t[k] for k in ("id", "worker", "instruction", "depends_on")} for t in tasks],
    )
    return {"tasks": tasks, "plan_summary": summary, "plan_source": source, "worker_results": {},
            "step_count": 0, "errors": errors}


# ---------------------------------------------------------------- supervisor


def route_next(state: AnalystState, max_steps: int, remaining_s: float) -> tuple[str, str | None, str, list[dict]]:
    """Pure routing decision: returns (next_node, task_id, reason, updated_tasks)."""
    tasks = [dict(t) for t in state.get("tasks", [])]
    by_id = {t["id"]: t for t in tasks}
    if state.get("step_count", 0) >= max_steps:
        return "synthesize", None, f"budget: {max_steps} supervisor dispatches used", tasks
    if remaining_s <= 0:
        return "synthesize", None, "budget: request time budget spent", tasks
    for t in tasks:
        if t["status"] == "failed" and t["attempts"] < MAX_ATTEMPTS and t.get("error_kind") in TRANSIENT:
            t["status"] = "pending"
            return t["worker"], t["id"], f"retry {t['id']}: transient {t['error_kind']} (attempt {t['attempts'] + 1})", tasks
    changed = True
    while changed:  # skipping can cascade down a dependency chain
        changed = False
        for t in tasks:
            if t["status"] == "pending" and any(by_id[d]["status"] in ("failed", "skipped") for d in t["depends_on"]):
                t["status"] = "skipped"
                changed = True
    for t in tasks:
        if t["status"] == "pending" and all(by_id[d]["status"] in DONE_LIKE for d in t["depends_on"]):
            why = f"after {', '.join(t['depends_on'])}" if t["depends_on"] else "no dependencies"
            return t["worker"], t["id"], f"task {t['id']} ready ({why})", tasks
    if not tasks:
        return "synthesize", None, "no tasks planned", tasks
    return "synthesize", None, "no pending tasks", tasks


async def supervisor(state: AnalystState, config) -> dict:
    ctx = ctx_of(config)
    step = state.get("step_count", 0)
    nxt, tid, reason, tasks = route_next(state, ctx.settings.max_supervisor_steps, ctx.remaining_s())
    if tid:
        step += 1
    tr = {"from": "supervisor", "to": nxt, "task_id": tid, "reason": reason, "step": step}
    ctx.emitter.emit("transition", **tr)
    for t in tasks:
        if t["status"] == "skipped" and not any(x["id"] == t["id"] and x["status"] == "skipped" for x in state["tasks"]):
            ctx.emitter.emit("worker_result", task_id=t["id"], agent=t["worker"], status="skipped",
                             error_kind="dependency_failed")
    return {"tasks": tasks, "next": nxt, "active_task": tid, "step_count": step, "transitions": [tr]}


# ---------------------------------------------------------------- workers


def classify_error(e: BaseException) -> str:
    if isinstance(e, (asyncio.TimeoutError, openai.APITimeoutError)):
        return "timeout"
    if isinstance(e, openai.RateLimitError):
        return "rate_limit"
    if isinstance(e, (openai.APIConnectionError, openai.InternalServerError)):
        return "provider_unavailable"
    return "worker_error"


def upstream_text(state: AnalystState, task: dict) -> str:
    parts = []
    for d in task["depends_on"]:
        r = state.get("worker_results", {}).get(d)
        if r:
            parts.append(f"[{d} {r.get('agent')} ({r['status']})]\n{r.get('findings_md') or '(no findings)'}")
    return "\n\n".join(parts)


def make_worker_node(name: str, fn: WorkerFn):
    async def node(state: AnalystState, config) -> dict:
        ctx = ctx_of(config)
        tasks = [dict(t) for t in state["tasks"]]
        task = next(t for t in tasks if t["id"] == state["active_task"])
        task["attempts"] += 1
        task["status"] = "running"
        errors = []
        timeout = max(1.0, min(ctx.settings.worker_timeout_s, ctx.remaining_s()))
        try:
            res = await asyncio.wait_for(fn(ctx, task, upstream_text(state, task)), timeout=timeout)
        except Exception as e:  # one worker failing must not take down the request
            kind = classify_error(e)
            msg = safe_error(e)
            res = {"status": "failed", "error_kind": kind, "error": msg, "findings_md": "", "evidence_ids": []}
            errors.append({"scope": name, "task_id": task["id"], "error_kind": kind, "message": msg, "ts": utc_now()})
            ctx.emitter.emit("error", scope="worker", agent=name, task_id=task["id"], message_safe=msg,
                             recoverable=True)
        res["agent"] = name
        res["task_id"] = task["id"]
        task["status"] = res["status"]
        task["error_kind"] = res.get("error_kind")
        ctx.emitter.emit(
            "worker_result",
            task_id=task["id"],
            agent=name,
            status=res["status"],
            error_kind=res.get("error_kind"),
            evidence_ids=res.get("evidence_ids", []),
            rounds=res.get("rounds"),
            tool_calls=res.get("tool_calls"),
        )
        ctx.emitter.emit("transition", **{"from": name, "to": "supervisor", "task_id": task["id"],
                                          "reason": f"{task['id']} {res['status']}", "step": state.get("step_count", 0)})
        return {"tasks": tasks, "worker_results": {**state.get("worker_results", {}), task["id"]: res},
                "errors": errors}

    node.__name__ = name
    return node


# ---------------------------------------------------------------- synthesize


def failure_answer(state: AnalystState) -> str:
    lines = ["I could not answer this question: every planned step failed, so there is no evidence to report."]
    for t in state.get("tasks", []):
        r = state.get("worker_results", {}).get(t["id"], {})
        lines.append(f"- {t['worker']} ({t['id']}): {t['status']}" + (f", {r.get('error_kind')}" if r.get("error_kind") else ""))
    lines.append("Please try again in a moment.")
    return "\n".join(lines)


async def synthesize(state: AnalystState, config) -> dict:
    ctx = ctx_of(config)
    tasks = state.get("tasks", [])
    results = state.get("worker_results", {})
    statuses = [t["status"] for t in tasks]

    if tasks and not any(s in DONE_LIKE for s in statuses):
        answer = failure_answer(state)
        ctx.emitter.emit("token", delta=answer)
        ctx.ttft_ms = ctx.ttft_ms or ctx.emitter.elapsed_ms()
        return {"final_answer": answer, "status": "failed"}

    if not tasks:
        role, refs, upstream = SYNTH_NO_TASK_ROLE, [], "none"
    else:
        role = SYNTH_ROLE
        ids: dict[str, None] = {}
        for t in tasks:
            for c in results.get(t["id"], {}).get("evidence_ids", []):
                ids.setdefault(c, None)
        refs = [ctx.registry.get(c) for c in ids] if ids else list(ctx.registry.by_cite.values())
        upstream = "\n\n".join(
            f"[{t['id']} {t['worker']} ({t['status']})]\n{results.get(t['id'], {}).get('findings_md') or '(no findings)'}"
            for t in tasks
        )
    inputs = {
        "worker_name": "Synthesizer",
        "role_paragraph": role,
        "retrieved_at": utc_now(),
        "question": state["query"],
        "instruction": "Write the final answer to the question.",
        "upstream": upstream,
        "evidence": format_evidence(refs),
    }
    features = {"n_evidence": len(refs), "n_tasks": len(tasks),
                "n_papers": len({r.doc_id for r in refs}), "statuses": statuses}
    decision = ctx.router.decide("synthesize", "synthesizer", features)
    ctx.emitter.emit("llm_route", **decision.public())
    chain = AUGMENT_PROMPT | ctx.router.client(decision.provider)

    t0 = time.perf_counter()
    first_ms, gathered, parts = None, None, []
    async for chunk in chain.astream(inputs):
        gathered = chunk if gathered is None else gathered + chunk
        delta = text_of(chunk.content)
        if delta:
            if first_ms is None:
                first_ms = int((time.perf_counter() - t0) * 1000)
                ctx.ttft_ms = ctx.emitter.elapsed_ms()
            parts.append(delta)
            ctx.emitter.emit("token", delta=delta)
    record_llm_result(ctx, decision, gathered, t0, ttft_ms=first_ms)
    answer = "".join(parts)

    used = cited_ids(answer)
    unresolved = [c for c in used if not ctx.registry.get(c)]
    for c in used:
        ref = ctx.registry.get(c)
        if ref:
            ctx.emitter.emit("citation", cite_id=c, type=ref.type, label=f"{ref.title} — {ref.section}",
                             doc_id=ref.doc_id, chunk_id=ref.chunk_id, href=ref.href, source_url=ref.source_url)
    status = "done" if all(s == "done" for s in statuses) else "partial"
    return {"final_answer": answer, "status": status, "unresolved_citations": unresolved}


# ---------------------------------------------------------------- graph


def build_graph():
    g = StateGraph(AnalystState)
    g.add_node("planner", planner)
    g.add_node("supervisor", supervisor)
    for name, (fn, _) in WORKERS.items():
        g.add_node(name, make_worker_node(name, fn))
        g.add_edge(name, "supervisor")
    g.add_node("synthesize", synthesize)
    g.add_edge(START, "planner")
    g.add_edge("planner", "supervisor")
    g.add_conditional_edges("supervisor", lambda s: s["next"], {**{n: n for n in WORKERS}, "synthesize": "synthesize"})
    g.add_edge("synthesize", END)
    return g.compile()


GRAPH = build_graph()


async def run_request(ctx: RequestContext) -> dict[str, Any]:
    """Run one request through the graph, emitting events; always ends with `done` and closes the stream."""
    em = ctx.emitter
    em.emit("meta", single_turn=True, k=ctx.k, workers=list(WORKERS))
    final: dict[str, Any] = {}
    try:
        final = await GRAPH.ainvoke(
            {"request_id": ctx.request_id, "query": ctx.query, "k": ctx.k},
            config={"configurable": {"ctx": ctx}, "recursion_limit": 40},
        )
        status = final.get("status", "failed")
    except asyncio.CancelledError:
        em.close()
        raise
    except Exception as e:
        status = "failed"
        em.emit("error", scope="request", message_safe=safe_error(e), recoverable=False)
    total_ms = em.elapsed_ms()
    em.emit(
        "done",
        status=status,
        plan_source=final.get("plan_source"),
        unresolved_citations=final.get("unresolved_citations", []),
        timings={
            "retrieval_ms": ctx.retrieval_ms_sum,
            "llm_ms_sum": ctx.llm_ms_sum,
            "ttft_ms": ctx.ttft_ms,
            "total_ms": total_ms,
        },
        tokens_by_provider=ctx.tokens_by_provider,
    )
    em.close()
    return final
