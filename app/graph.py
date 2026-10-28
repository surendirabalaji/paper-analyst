"""LangGraph supervisor graph.

    START -> planner -> supervisor --route_next--> research | summary | data | synthesize
                         ^              |
                         +--- worker ---+          synthesize -> END

The planner is one structured-output LLM call that produces a small task list. The supervisor is
deterministic: `route_next` reads task statuses, dependencies, retry counts and the request budget,
and every decision is emitted as a `transition` event with its reason. Workers always return to the
supervisor, and an exception or timeout inside a worker becomes a recorded `failed` result.
Workers: Research (Phase 2), Summary (Phase 5) and Data (Phase 6). A task that depends on others gets
their findings and, as `upstream_papers`, the paper ids they found (traced as a `handoff` event).

Failure handling (Phase 7):
- Task status is pending | running | done | partial | failed | skipped. `partial` counts as usable evidence
  (dependents still run); a failed or skipped dependency skips its dependents with a stated reason.
- Transient failures (timeout, rate limit, provider unavailable, empty response) are retried up to
  MAX_ATTEMPTS, unless both generation providers already failed in this request or no time is left;
  other failures are not retried. Every retry, and every retry not made, is a `transition` with its reason.
- Time: workers run inside the request budget minus a reserve kept for synthesis (`worker_window_s`), and
  every LLM attempt is capped by the time left (router.attempt_timeout), so a hung call cannot overrun
  REQUEST_BUDGET_S (P2-N7 / P4-N1).
- Synthesis lists what is missing and why (a deterministic "Incomplete evidence" note follows the streamed
  answer). If every task failed, a templated answer says what was tried and whether retrying can help (N9).
  If the synthesis call itself fails before streaming, the workers' cited findings are returned as-is.
- Fault injection (app/faults.py, `worker:<name>:<mode>`) is applied here; injected failures are labelled.

Latency (Phase 8b, package B; each part can be switched off in config):
- The planner also names Summary's papers and Research's search queries. They are checked (papers must
  resolve in papers.db) and become the task's `seeds`: tool calls the worker runs before its first LLM round.
- While the planner plans, a corpus search of the question runs speculatively. It is kept as a Research seed
  only when the plan is a single Research task, and discarded otherwise; both are traced.
- One-task plans run the Research or Summary worker in evidence-only mode (no written findings); the
  synthesizer answers from the evidence it gathered. Plans with two or more tasks keep worker findings,
  because the synthesizer must reconcile them.
"""

from __future__ import annotations

import asyncio
import operator
import re
import time
from typing import Annotated, Any, Awaitable, Callable, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.agents.data import run_data
from app.agents.prompts import PLANNER_SYSTEM, SYNTH_NO_TASK_ROLE, SYNTH_ROLE
from app.agents.research import run_research
from app.agents.summary import PaperNotFound, resolve_paper, run_summary
from app.context import RequestContext, ctx_of
from app.events import safe_error, utc_now
from app.faults import is_injected, maybe_fail
from app.llm.policy import approx_tokens, comparison_cue
from app.llm.router import call_llm, classify_error, stream_llm
from app.rag import AUGMENT_PROMPT, cited_ids, format_evidence

MAX_TASKS = 4
MAX_ATTEMPTS = 2
TRANSIENT = {"timeout", "rate_limit", "provider_unavailable", "empty_response"}
DONE_LIKE = {"done", "partial"}
MIN_WORKER_S = 5.0  # do not start (or retry) a worker with less time than this left in its window
BUDGET_SKIPPED = "budget"
# Plain-language reason per error kind, used in skip reasons, the "Incomplete evidence" note and the
# all-failed answer. RETRYABLE kinds may succeed if the user simply asks again; the others will not.
REASONS = {
    "timeout": "timed out",
    "rate_limit": "was rate-limited by the model provider",
    "provider_unavailable": "could not reach a model provider",
    "empty_response": "got an empty response from the model provider",
    "budget_exhausted": "ran out of request time",
    BUDGET_SKIPPED: "was not run because the request time or step budget ran out",
    "worker_error": "failed with an internal error",
    "tool_error": "had a tool failure, so its results are incomplete",
    "live_search_unavailable": "could not use live web search, so current information was not checked",
    "paper_not_found": "could not find a named paper in the indexed corpus",
    "invalid_sql": "wrote a database query that was still invalid after one repair",
    "sql_rejected": "wrote a database query that the read-only SQL policy refused",
    "sql_timeout": "wrote a database query that exceeded the time limit",
    "data_unavailable": "could not open the corpus database",
    "empty_result": "returned no findings",
    "citation_or_empty": "returned findings without resolvable citations",
    "no_query": "did not run a database query",
}
RETRYABLE = TRANSIENT | {"budget_exhausted", BUDGET_SKIPPED, "live_search_unavailable"}
SERVICE_SIDE = {"data_unavailable", "worker_error", "tool_error"}  # neither retrying nor rephrasing helps

WorkerFn = Callable[[RequestContext, dict, str], Awaitable[dict]]
SPEC_MIN_CHARS, SPEC_MAX_CHARS = 2, 300  # corpus_search's query limits: only a valid query is searched speculatively
EVIDENCE_ONLY_WORKERS = ("research", "summary")  # Data keeps its findings: its SQL result needs explaining
WORKERS: dict[str, tuple[WorkerFn, str]] = {
    "research": (
        run_research,
        "research: answers conceptual and cross-paper questions by vector search over the full text of the "
        "indexed papers, with citations; with needs_live it also searches the live web (current developments, "
        "or topics the corpus does not cover).",
    ),
    "summary": (
        run_summary,
        "summary: summarises or explains one or two specific named papers section by section, with citations; "
        "with needs_live it also checks a paper's current status on the web (venue, acceptance, follow-ups).",
    ),
    "data": (
        run_data,
        "data: exact metadata about the indexed papers from the corpus database (read-only SQL): counts, "
        "lists, publication dates and years, authors, arXiv categories, topics, which papers have a code "
        "repository. Not for what the papers say.",
    ),
}


class AnalystState(TypedDict, total=False):
    request_id: str
    query: str
    k: int
    plan_summary: str
    plan_source: str  # "llm" or "fallback"
    tasks: list[dict]  # {id, worker, instruction, needs_live, depends_on, status, attempts, error_kind, reason}
    worker_results: dict[str, dict]
    next: str
    active_task: str | None
    step_count: int
    transitions: Annotated[list[dict], operator.add]
    errors: Annotated[list[dict], operator.add]
    final_answer: str
    status: str
    unresolved_citations: list[str]
    dropped_citations: list[str]
    excluded_evidence: list[str]
    gaps: list[dict]  # tasks whose evidence is missing or incomplete, with reasons (Phase 7)


# ---------------------------------------------------------------- planner


class PlannedTask(BaseModel):
    worker: str = Field(..., description="Name of an available worker")
    instruction: str = Field(..., description="What this worker should find or produce")
    depends_on: list[int] = Field(default_factory=list, description="0-based indices of earlier tasks it needs")
    needs_live: bool = Field(False, description="True if the task needs current information from the live web")
    papers: list[str] = Field(default_factory=list, description="summary only: arXiv ids or short titles of the "
                              "named papers (at most 2)")
    queries: list[str] = Field(default_factory=list, description="research only: 1-2 short corpus search queries")


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
                "needs_live": bool(t.needs_live),
                "depends_on": deps,
                "status": "pending",
                "attempts": 0,
                "error_kind": None,
                # Seed hints (Phase 8b): only for the worker that can use them, trimmed and capped.
                "papers": _hints(t.papers, 2, 200) if t.worker == "summary" else [],
                "queries": _hints(t.queries, 2, 300) if t.worker == "research" else [],
            }
        )
    return tasks, notes


def _hints(values: list[str], cap: int, max_chars: int) -> list[str]:
    out: list[str] = []
    for v in values or []:
        v = (v or "").strip()[:max_chars]
        if len(v) >= 3 and v not in out:
            out.append(v)
    return out[:cap]


async def plan_seeds(ctx: RequestContext, tasks: list[dict], notes: list[str]) -> None:
    """Turn the planner's hints into each task's `seeds` (tool calls run before the worker's first LLM
    round), and keep or discard the speculative search of the question. Unknown papers are dropped with
    a note; the worker then finds the paper itself and reports it if it is not in the corpus."""
    single_research = len(tasks) == 1 and tasks[0]["worker"] == "research"
    use_spec = bool(ctx.speculative) and single_research
    for t in tasks:
        seeds: list[dict] = []
        if t["worker"] == "research":
            if use_spec:
                seeds.append({"name": "corpus_search", "args": {"query": ctx.query}, "by": "speculative"})
            if ctx.settings.planner_seeds:
                for q in t.get("queries", [])[: 1 if seeds else 2]:
                    if q != ctx.query or not seeds:
                        seeds.append({"name": "corpus_search", "args": {"query": q}, "by": "planner"})
        elif t["worker"] == "summary" and ctx.settings.planner_seeds:
            for ref in t.get("papers", []):
                try:
                    row = await asyncio.to_thread(resolve_paper, ctx.settings.sqlite_path, ref)
                except PaperNotFound as e:
                    notes.append(f"{t['id']}: planner-named paper not seeded ({str(e)[:120]})")
                    continue
                except Exception as e:  # the database is the Data/Summary tools' problem to report, not ours
                    notes.append(f"{t['id']}: paper seed skipped ({safe_error(e)[:120]})")
                    continue
                if all(s["args"]["paper"] != row["arxiv_id"] for s in seeds):
                    seeds.append({"name": "get_paper_outline", "args": {"paper": row["arxiv_id"]}, "by": "planner"})
        t["seeds"] = seeds
    if ctx.speculative:
        await ctx.end_speculative(use_spec, "plan is a single Research task: kept as its first corpus search" if use_spec
                            else "plan is not a single Research task")


def fallback_tasks(query: str) -> list[dict]:
    return [
        {"id": "t1", "worker": "research", "instruction": query, "needs_live": True, "depends_on": [],
         "status": "pending", "attempts": 0, "error_kind": None}
    ]


async def planner(state: AnalystState, config) -> dict:
    ctx = ctx_of(config)
    ctx.emitter.emit("transition", **{"from": "start", "to": "planner", "reason": "new request", "step": 0})
    decision = ctx.route("planner", "planner", {"query_chars": len(state["query"]),
                                                "comparison": comparison_cue(state["query"])})
    system = PLANNER_SYSTEM.format(max_tasks=MAX_TASKS, workers="\n".join(f"- {d}" for _, d in WORKERS.values()))
    errors, source = [], "llm"
    if ctx.settings.speculative_retrieval and SPEC_MIN_CHARS <= len(state["query"].strip()) <= SPEC_MAX_CHARS:
        ctx.start_speculative(state["query"], ctx.k)
    ctx.stage_deadline = time.perf_counter() + max(0.0, ctx.worker_window_s())  # keep the synthesis reserve
    try:
        out = await call_llm(
            ctx, decision,
            lambda c: c.with_structured_output(Plan, method="function_calling", include_raw=True),
            [SystemMessage(system), HumanMessage(state["query"])],
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
        ctx.emitter.emit("error", scope="planner", error_kind="planner_error", message_safe=msg, recoverable=True,
                         **({"injected": True} if is_injected(e) else {}))
    finally:
        ctx.stage_deadline = None
    await plan_seeds(ctx, tasks, notes)
    ctx.emitter.emit(
        "plan",
        plan_summary=summary,
        source=source,
        notes=notes,
        tasks=[{**{k: t[k] for k in ("id", "worker", "instruction", "needs_live", "depends_on")},
                "seeds": [{"tool": x["name"], **x["args"], "by": x["by"]} for x in t.get("seeds", [])]}
               for t in tasks],
    )
    return {"tasks": tasks, "plan_summary": summary, "plan_source": source, "worker_results": {},
            "step_count": 0, "errors": errors}


# ---------------------------------------------------------------- supervisor


NO_UPSTREAM_PAPERS = "no_upstream_papers"


def found_no_papers(task: dict, by_id: dict, results: dict) -> bool:
    """True when every dependency is a Data task whose queries succeeded and returned zero rows. Rows without
    an arxiv_id column (e.g. titles only) still count as found: the dependent task gets them through the
    upstream findings (tester P6-D2)."""
    deps = task["depends_on"]
    return bool(deps) and all(
        by_id[d]["worker"] == "data" and (results.get(d) or {}).get("sql_ok")
        and not (results.get(d) or {}).get("sql_rows") and not (results.get(d) or {}).get("paper_ids")
        for d in deps)


def _skip_unrun(tasks: list[dict], why: str) -> None:
    for t in tasks:
        if t["status"] == "pending":
            t["status"], t["error_kind"], t["reason"] = "skipped", BUDGET_SKIPPED, why


def route_next(state: AnalystState, max_steps: int, remaining_s: float, *, min_worker_s: float = 0.0,
               providers_down: bool = False) -> tuple[str, str | None, str, list[dict]]:
    """Pure routing decision: returns (next_node, task_id, reason, updated_tasks).

    `remaining_s` is the time a worker could still use (the request budget minus the synthesis reserve);
    `providers_down` means both generation providers already failed in this request."""
    tasks = [dict(t) for t in state.get("tasks", [])]
    by_id = {t["id"]: t for t in tasks}
    if state.get("step_count", 0) >= max_steps:
        why = f"budget: {max_steps} supervisor dispatches used"
        _skip_unrun(tasks, f"the step budget ({max_steps} dispatches) ran out before it could run")
        return "synthesize", None, why, tasks
    if remaining_s <= min_worker_s:
        why = (f"budget: request time budget spent ({max(remaining_s, 0):.1f} s left for workers; "
               "the rest is kept for the answer)")
        _skip_unrun(tasks, "the request time budget ran out before it could run")
        return "synthesize", None, why, tasks
    notes = []
    for t in tasks:
        if t["status"] == "failed" and t.get("error_kind") in TRANSIENT and not t.get("retry_note"):
            if t["attempts"] >= MAX_ATTEMPTS:
                t["retry_note"] = f"not retrying {t['id']}: {MAX_ATTEMPTS} attempts used"
            elif providers_down:
                t["retry_note"] = f"not retrying {t['id']}: both generation providers already failed in this request"
            else:
                t["status"] = "pending"
                return (t["worker"], t["id"],
                        f"retry {t['id']}: transient {t['error_kind']} (attempt {t['attempts'] + 1} of {MAX_ATTEMPTS})", tasks)
            notes.append(t["retry_note"])
    changed = True
    while changed:  # skipping can cascade down a dependency chain
        changed = False
        for t in tasks:
            bad = [by_id[d] for d in t["depends_on"] if by_id[d]["status"] in ("failed", "skipped")]
            if t["status"] == "pending" and bad:
                d = bad[0]
                t["status"], t["error_kind"] = "skipped", "dependency_failed"
                dep_why = (REASONS.get(d.get("error_kind")) if d["status"] == "failed" else None) or (
                    "was skipped" if d["status"] == "skipped" else d["status"])
                t["reason"] = f"it needs {d['id']} ({d['worker']}), which {dep_why}"
                changed = True
    results = state.get("worker_results", {})
    for t in tasks:
        if t["status"] == "pending" and all(by_id[d]["status"] in DONE_LIKE for d in t["depends_on"]):
            if t["worker"] == "summary" and found_no_papers(t, by_id, results):
                # Data answered "no such papers": there is nothing to summarise, and that is not a failure.
                t["status"], t["error_kind"] = "skipped", NO_UPSTREAM_PAPERS
                continue
            why = f"after {', '.join(t['depends_on'])}" if t["depends_on"] else "no dependencies"
            return t["worker"], t["id"], "; ".join([*notes, f"task {t['id']} ready ({why})"]), tasks
    if not tasks:
        return "synthesize", None, "no tasks planned", tasks
    return "synthesize", None, "; ".join([*notes, "no pending tasks"]), tasks


async def supervisor(state: AnalystState, config) -> dict:
    ctx = ctx_of(config)
    step = state.get("step_count", 0)
    nxt, tid, reason, tasks = route_next(state, ctx.settings.max_supervisor_steps, ctx.worker_window_s(),
                                         min_worker_s=MIN_WORKER_S, providers_down=ctx.providers_down())
    if tid:
        step += 1
    tr = {"from": "supervisor", "to": nxt, "task_id": tid, "reason": reason, "step": step}
    ctx.emitter.emit("transition", **tr)
    for t in tasks:
        if t["status"] == "skipped" and not any(x["id"] == t["id"] and x["status"] == "skipped" for x in state["tasks"]):
            ctx.emitter.emit("worker_result", task_id=t["id"], agent=t["worker"], status="skipped",
                             error_kind=t.get("error_kind") or "dependency_failed", reason=t.get("reason"))
    return {"tasks": tasks, "next": nxt, "active_task": tid, "step_count": step, "transitions": [tr]}


# ---------------------------------------------------------------- workers


def upstream_text(state: AnalystState, task: dict) -> str:
    parts = []
    for d in task["depends_on"]:
        r = state.get("worker_results", {}).get(d)
        if r:
            parts.append(f"[{d} {r.get('agent')} ({r['status']})]\n{r.get('findings_md') or '(no findings)'}")
    return "\n\n".join(parts)


def upstream_papers(state: AnalystState, task: dict) -> list[dict]:
    """Paper ids found by the tasks this one depends on (Data's `paper_ids`), in dependency order."""
    seen, out = set(), []
    for d in task["depends_on"]:
        r = state.get("worker_results", {}).get(d) or {}
        if r.get("status") not in DONE_LIKE:
            continue
        for p in r.get("paper_ids") or []:
            if p["arxiv_id"] not in seen:
                seen.add(p["arxiv_id"])
                out.append({**p, "from_task": d})
    return out


def make_worker_node(name: str, fn: WorkerFn):
    async def node(state: AnalystState, config) -> dict:
        ctx = ctx_of(config)
        tasks = [dict(t) for t in state["tasks"]]
        task = next(t for t in tasks if t["id"] == state["active_task"])
        task["attempts"] += 1
        task["status"] = "running"
        # One-task plans: the worker gathers evidence and the synthesizer writes from it (Phase 8b).
        task["evidence_only"] = (ctx.settings.evidence_only_single_task and len(tasks) == 1
                                 and name in EVIDENCE_ONLY_WORKERS)
        errors = []
        if task["depends_on"]:
            papers = upstream_papers(state, task)
            task["upstream_papers"] = papers
            ctx.emitter.emit("handoff", to_task=task["id"], to_agent=name, from_tasks=task["depends_on"],
                             papers=[{k: p.get(k) for k in ("arxiv_id", "title", "cite_id", "from_task")} for p in papers],
                             evidence_ids=[c for d in task["depends_on"]
                                           for c in (state.get("worker_results", {}).get(d) or {}).get("evidence_ids", [])])
        # The worker's window: its own timeout, but never past the time kept back for synthesis. The router
        # caps each LLM attempt by ctx.time_left(), so the stage deadline is enforced inside the worker too.
        timeout = max(1.0, min(ctx.settings.worker_timeout_s, ctx.worker_window_s()))
        hits0 = len(ctx.faults.hits)
        real0 = len(ctx.real_provider_errors)

        async def run() -> dict:
            if await maybe_fail(ctx, "worker", name, task_id=task["id"], attempt=task["attempts"]) == "empty":
                return {"status": "failed", "error_kind": "empty_result", "findings_md": "", "evidence_ids": [],
                        "error": "[injected test fault] the worker returned no findings"}
            return await fn(ctx, task, upstream_text(state, task))

        ctx.stage_deadline = time.perf_counter() + timeout
        try:
            # The extra second lets the capped LLM attempt time out first, so its llm_result says why.
            res = await asyncio.wait_for(run(), timeout=timeout + 1.0)
        except Exception as e:  # one worker failing must not take down the request
            kind = classify_error(e)
            msg = safe_error(e)
            res = {"status": "failed", "error_kind": kind, "error": msg, "findings_md": "", "evidence_ids": []}
            errors.append({"scope": name, "task_id": task["id"], "error_kind": kind, "message": msg, "ts": utc_now(),
                           "injected": is_injected(e)})
            ctx.emitter.emit("error", scope="worker", agent=name, task_id=task["id"], error_kind=kind,
                             message_safe=msg, recoverable=True, **({"injected": True} if is_injected(e) else {}))
        finally:
            ctx.stage_deadline = None
        faults = [h["target"] for h in ctx.faults.hits[hits0:]]
        real = ctx.real_provider_errors[real0:]
        if faults:
            res["faults"] = faults
            # Caused (or shaped) by injected faults only if no real provider error happened in this worker too
            # (P8b-D2): a real outage during a fault test is reported as real, with the fault noted.
            res["injected"] = res["status"] != "done" and not real
            if res["status"] != "done" and real:
                res["real_errors"] = real
        res["agent"] = name
        res["task_id"] = task["id"]
        task["status"] = res["status"]
        task["error_kind"] = res.get("error_kind")
        task["injected"] = bool(res.get("injected"))
        task["fault_also"] = bool(res.get("real_errors"))  # real failure while a test fault was also active
        task.pop("retry_note", None)
        ctx.emitter.emit(
            "worker_result",
            task_id=task["id"],
            agent=name,
            status=res["status"],
            error_kind=res.get("error_kind"),
            attempt=task["attempts"],
            evidence_ids=res.get("evidence_ids", []),
            rounds=res.get("rounds"),
            tool_calls=res.get("tool_calls"),
            **({"evidence_only": True} if task["evidence_only"] else {}),
            **({"injected": True, "faults": faults} if res.get("injected") else {}),
            **({"real_errors": res["real_errors"], "faults": faults} if res.get("real_errors") else {}),
        )
        ctx.emitter.emit("transition", **{"from": name, "to": "supervisor", "task_id": task["id"],
                                          "reason": f"{task['id']} {res['status']}", "step": state.get("step_count", 0)})
        return {"tasks": tasks, "worker_results": {**state.get("worker_results", {}), task["id"]: res},
                "errors": errors}

    node.__name__ = name
    return node


# ---------------------------------------------------------------- synthesize


LABELS = {"research": "Research", "summary": "Summary", "data": "Data"}
EVIDENCE_ONLY_NOTE = ("(evidence-only task: the worker gathered the <evidence> items below and wrote no findings; "
                      "answer this task directly from them)")


def describe(task: dict) -> str:
    """One line: who, what happened and why (with the injected label when a test fault caused it)."""
    kind, status = task.get("error_kind"), task["status"]
    why = REASONS.get(kind) or (kind or "").replace("_", " ")
    who = f"{LABELS.get(task['worker'], task['worker'])} ({task['id']})"
    if status == "skipped":
        line = f"{who} was skipped: {task['reason']}" if task.get("reason") else f"{who} {why or 'was skipped'}"
    elif status == "partial":
        line = f"{who} is incomplete" + (f": it {why}" if why else "")
    elif status == "failed":
        line = f"{who} {why or 'failed'}"
    else:
        line = f"{who} {status}"
    if task.get("injected"):
        return line + " [injected test fault, not a real outage]"
    if task.get("fault_also"):
        return line + " [a real provider error, not caused by the injected test fault that was also active]"
    return line


def evidence_gaps(tasks: list[dict]) -> list[dict]:
    """Tasks whose evidence is missing (failed / skipped) or incomplete (partial), in plan order. A Summary
    skipped because Data found no papers is a complete answer, not a gap."""
    return [{"task_id": t["id"], "worker": t["worker"], "status": t["status"], "error_kind": t.get("error_kind"),
             "injected": bool(t.get("injected")), "text": describe(t)}
            for t in tasks
            if t["status"] in ("failed", "skipped", "partial") and t.get("error_kind") != NO_UPSTREAM_PAPERS]


def gaps_note(gaps: list[dict]) -> str:
    if not gaps:
        return ""
    return "\n\n---\n**Incomplete evidence:** " + "; ".join(g["text"] for g in gaps) + "."


def failure_answer(state: AnalystState) -> str:
    """Templated answer when no task produced usable evidence (no LLM call). The closing advice matches the
    cause (N9): injected faults are named as such, transient problems invite a retry, deterministic ones
    (a refused query, a paper not in the corpus) say that retrying will not help."""
    tasks = state.get("tasks", [])
    lines = ["I could not answer this question: no planned step produced usable evidence, so there is nothing "
             "grounded to report. What was tried:"]
    lines += [f"- {describe(t)}" for t in tasks]
    root = [t for t in tasks if t["status"] == "failed"] or tasks  # skipped tasks follow from these
    real = [t for t in root if not t.get("injected")]
    if any(t.get("injected") for t in root):
        lines.append("These failures were injected on purpose to test failure handling (fault injection); "
                     "they do not indicate a real outage." if not real else
                     "The failures marked as injected test faults were caused on purpose (fault injection); "
                     "the others are real and are described below.")
    if real and all(t.get("error_kind") in RETRYABLE for t in real):
        lines.append("This looks temporary (a provider, network or time-limit problem). Please try again in a moment.")
    elif real:
        first = next(t for t in real if t.get("error_kind") not in RETRYABLE)
        what = (f"{LABELS.get(first['worker'], first['worker'])} "
                f"{REASONS.get(first.get('error_kind'), first.get('error_kind') or 'failed')}")
        if first.get("error_kind") in SERVICE_SIDE:
            lines.append(f"This is a problem on the service side ({what}), not with your question, so neither "
                         "retrying nor rephrasing is likely to help until it is fixed.")
        else:
            lines.append(f"Asking the same question again will probably not help: {what}. Try rephrasing it, "
                         "naming the paper by its arXiv id, or asking for one thing at a time.")
    return "\n".join(lines)


def fallback_findings_answer(tasks: list[dict], results: dict, err: str) -> str:
    """Synthesis failed before streaming anything: return the workers' own cited findings, labelled."""
    parts = [f"The final write-up step failed ({err}), so these are the research findings as the workers "
             "returned them, with their citations:"]
    for t in tasks:
        f = (results.get(t["id"]) or {}).get("findings_md") or ""
        if t["status"] in DONE_LIKE and f.strip():
            parts.append(f"**{LABELS.get(t['worker'], t['worker'])} ({t['id']})**\n{f.strip()}")
    return "\n\n".join(parts)


async def synthesize(state: AnalystState, config) -> dict:
    ctx = ctx_of(config)
    tasks = state.get("tasks", [])
    results = state.get("worker_results", {})
    statuses = [t["status"] for t in tasks]

    gaps = evidence_gaps(tasks)
    if tasks and not any(s in DONE_LIKE for s in statuses):
        answer = failure_answer(state)
        ctx.emitter.emit("token", delta=answer)
        ctx.ttft_ms = ctx.ttft_ms or ctx.emitter.elapsed_ms()
        return {"final_answer": answer, "status": "failed", "gaps": gaps}

    off_ids: set[str] = set()
    if not tasks:
        role, refs, upstream = SYNTH_NO_TASK_ROLE, [], "none"
    else:
        role = SYNTH_ROLE
        ids: dict[str, None] = {}
        for t in tasks:
            for c in results.get(t["id"], {}).get("evidence_ids", []):
                ids.setdefault(c, None)
        # Only evidence the workers actually cited. A worker that abstained cited nothing, so synthesis gets
        # no evidence rather than every retrieved (and possibly unrelated) chunk (P3-N3). Corpus chunks with
        # off-topic similarity are dropped too: a worker cites them only to say the corpus lacks the answer
        # (P5-D1). They are never linked and are reported in done.dropped_citations.
        off_ids = {c for c in ids if ctx.registry.corpus_off_topic(c, LOW_RELEVANCE_SCORE)}
        refs = [ctx.registry.get(c) for c in ids if c not in off_ids]
        upstream = "\n\n".join(
            f"[{t['id']} {t['worker']} ({t['status']})]\n" + (
                "(not run: the database lookup found no matching papers, so there was nothing to summarise)"
                if t.get("error_kind") == NO_UPSTREAM_PAPERS
                else EVIDENCE_ONLY_NOTE if results.get(t["id"], {}).get("evidence_only")
                else results.get(t["id"], {}).get("findings_md") or "(no findings)")
            for t in tasks
        )
        if gaps:
            upstream += ("\n\nMissing or incomplete evidence (a note listing this is added after your answer; "
                         "answer only what the evidence supports and do not fill these gaps):\n"
                         + "\n".join(f"- {g['text']}" for g in gaps))
    inputs = {
        "worker_name": "Synthesizer",
        "role_paragraph": role,
        "retrieved_at": utc_now(),
        "question": state["query"],
        "instruction": "Write the final answer to the question.",
        "upstream": upstream,
        "evidence": format_evidence(refs),
    }
    features = {
        "n_tasks": len(tasks),
        "n_tasks_ok": sum(1 for x in statuses if x in DONE_LIKE),
        "n_evidence": len(refs),
        "n_papers": len({r.doc_id for r in refs if r.type == "paper_chunk"}),
        "n_web": sum(1 for r in refs if r.type == "web"),
        "n_data": sum(1 for r in refs if r.type in ("sql_row", "sql_result")),
        "evidence_tokens_est": sum(approx_tokens(r.text) for r in refs),
        "comparison": comparison_cue(state["query"]),
        "statuses": statuses,
    }
    if any(results.get(t["id"], {}).get("evidence_only") for t in tasks):
        features["evidence_only"] = True  # answers from gathered evidence, not worker findings (Phase 8b)
    decision = ctx.route("synthesize" if tasks else "synthesize_no_task", "synthesizer", features)

    streamed = {"n": 0}

    def on_delta(delta: str) -> None:
        if ctx.ttft_ms is None:
            ctx.ttft_ms = ctx.emitter.elapsed_ms()
        streamed["n"] += 1
        ctx.emitter.emit("token", delta=delta)

    synth_failed = False
    try:
        answer = await stream_llm(ctx, decision, lambda c: AUGMENT_PROMPT | c, inputs, on_delta)
    except Exception as e:
        if streamed["n"] or not tasks:  # part of the answer is already on screen: report the break as an error
            raise
        # Nothing streamed yet (both providers failed, or no time left): the workers' own cited findings
        # are still grounded, so return them rather than nothing.
        synth_failed = True
        msg = safe_error(e)
        ctx.emitter.emit("error", scope="synthesize", error_kind=classify_error(e), message_safe=msg,
                         recoverable=True, **({"injected": True} if is_injected(e) else {}))
        answer = fallback_findings_answer(tasks, results, f"{classify_error(e)}"
                                          + (", injected test fault" if is_injected(e) else ""))
        on_delta(answer)
    note = gaps_note(gaps)
    if note:  # always say what is missing, whatever the model wrote
        on_delta(note)

    used = cited_ids(answer)
    unresolved = [c for c in used if not ctx.registry.get(c)]
    dropped = abstention_only_markers(answer) | (set(used) & off_ids)
    for c in used:
        ref = ctx.registry.get(c)
        if ref and c not in dropped:
            ctx.emitter.emit("citation", cite_id=c, type=ref.type, label=f"{ref.title} — {ref.section}",
                             title=ref.title, section=ref.section,
                             **({"snippet": ref.text} if ref.type == "web" else {}),
                             doc_id=ref.doc_id, chunk_id=ref.chunk_id, href=ref.href, source_url=ref.source_url,
                             retrieved_at=ref.retrieved_at or None, found_by=ref.agent or None,
                             **({"sql": ref.sql, "columns": ref.columns, "rows": ref.rows}
                                if ref.type in ("sql_row", "sql_result") else {}))
    # A Summary skipped because Data found no papers is a complete answer ("no such papers"), not a gap.
    status = "done" if not synth_failed and all(t["status"] == "done" or t.get("error_kind") == NO_UPSTREAM_PAPERS
                                                for t in tasks) else "partial"
    return {"final_answer": answer + note, "status": status, "unresolved_citations": unresolved,
            "dropped_citations": [c for c in used if c in dropped], "excluded_evidence": sorted(off_ids),
            "gaps": gaps}


# A statement that the *evidence* (papers, corpus, passages, sources) lacks something. The subject is required,
# so a supported claim such as "standard RAG cannot address noisy passages [C2]" is not an abstention (P5-N3).
_SOURCES = r"(evidence|papers?|corpus|passages?|sources?|documents?|excerpts?|context|literature|results)"
_ABSTAIN = re.compile(
    _SOURCES + r"\b[^.;]{0,40}?\b(does not|doesn't|do not|don't|cannot|can't|did not|fail to)\s+"
    r"(answer|cover|address|contain|include|mention|say|describe|discuss|support|provide|report)"
    r"|not (covered|found|available|addressed|mentioned|reported) (in|by) (the )?" + _SOURCES
    + r"|no (relevant |such )?(evidence|information|data) (was |is )?(found|available|retrieved)|no relevant evidence"
    r"|evidence is insufficient|insufficient evidence|not in the (corpus|indexed papers|evidence)",
    re.I,
)
_CLAUSE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+|(;)\s*|,?\s+but\s+", re.I)
_CONTINUES = re.compile(r"^\s*(thus|so|therefore|hence|consequently|as a result|instead)\b", re.I)


# Applies only to chunks found solely by corpus-wide search (off-topic questions score 0.16-0.26 there,
# in-domain top-k >= 0.58). Chunks Summary read or searched inside a named paper are never filtered: scoped
# keyword searches legitimately score 0.20-0.45 (tester P5-D3).
LOW_RELEVANCE_SCORE = 0.35


def abstention_only_markers(answer: str) -> set[str]:
    """Markers that appear only in clauses saying the evidence does not answer something. They are not
    linked as citations (P3-N3): a statement of missing evidence is not supported by the passage.
    A clause after ';' that continues an abstention ("…do not contain X; thus, figures come from the web
    [C1], [W1]") inherits it for paper markers (C#); web markers there still count as support (P5-D1)."""
    support: dict[str, bool] = {}
    parts = _CLAUSE_SPLIT.split(answer or "")
    prev_abstains, after_semicolon = False, False
    for part in parts:
        if part is None:
            continue
        if part == ";":
            after_semicolon = True
            continue
        abstains = bool(_ABSTAIN.search(part))
        inherited = after_semicolon and prev_abstains and bool(_CONTINUES.match(part))
        for c in cited_ids(part):
            # A D# is a database result; an empty one is what supports "the corpus has no such papers".
            unsupported = (abstains or (inherited and c.startswith("C"))) and not c.startswith("D")
            support[c] = support.get(c, False) or not unsupported
        if part.strip():
            prev_abstains, after_semicolon = abstains, False
    return {c for c, ok in support.items() if not ok}


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
    em.emit("meta", single_turn=True, k=ctx.k, workers=list(WORKERS), budget_s=ctx.settings.request_budget_s,
            **({"fault_injection": {"rules": ctx.faults.spec(), "source": ctx.faults.source,
                                    "note": "Test faults are injected in this request; failures they cause are labelled."}}
               if ctx.faults.active else {}))
    final: dict[str, Any] = {}
    try:
        final = await GRAPH.ainvoke(
            {"request_id": ctx.request_id, "query": ctx.query, "k": ctx.k},
            config={"configurable": {"ctx": ctx}, "recursion_limit": 40},
        )
        status = final.get("status", "failed")
    except asyncio.CancelledError:
        if ctx.speculative:
            ctx.speculative["task"].cancel()
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
        dropped_citations=final.get("dropped_citations", []),
        excluded_evidence=final.get("excluded_evidence", []),  # worker-cited but off-topic, withheld from synthesis
        tasks=[{"id": t["id"], "worker": t["worker"], "status": t["status"], "attempts": t.get("attempts", 0),
                "error_kind": t.get("error_kind"), "reason": t.get("reason"), "injected": bool(t.get("injected"))}
               for t in final.get("tasks", [])],
        gaps=final.get("gaps", []),
        faults_injected=ctx.faults.hits,
        over_budget_s=round(max(0.0, -ctx.remaining_s()), 2),
        timings={
            "retrieval_ms": ctx.retrieval_ms_sum,
            "llm_ms_sum": ctx.llm_ms_sum,
            "ttft_ms": ctx.ttft_ms,
            "total_ms": total_ms,
        },
        tokens_by_provider=ctx.tokens_by_provider,
        providers_used=sorted(p for p, v in ctx.tokens_by_provider.items() if v["calls"] > v["errors"]),
        fallbacks=ctx.fallbacks,
        cost_usd=round(sum(v.get("cost_usd", 0.0) for v in ctx.tokens_by_provider.values()), 6),
        cost_basis=ctx.cost_basis,
    )
    em.close()
    return final
