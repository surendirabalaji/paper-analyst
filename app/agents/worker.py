"""Generic bounded tool-calling loop used by every worker.

Bounds: at most `max_rounds` LLM rounds, at most MAX_CALLS_PER_ROUND tool calls per round and
MAX_CALLS_PER_TASK per task (each result stays in the context, so this also bounds prompt size and cost),
and a final tool-free round if the budget runs out. Every LLM call is routed and traced; every tool
call is traced as tool_call / tool_result with the calling agent's identity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from app.context import RequestContext
from app.llm.policy import approx_tokens, comparison_cue
from app.llm.router import call_llm, text_of

MAX_CALLS_PER_ROUND = 3
MAX_CALLS_PER_TASK = 4
SAFE_ARG_CHARS = 200


@dataclass
class LoopOutput:
    text: str
    rounds: int
    tool_calls: int = 0
    tool_errors: list[str] = field(default_factory=list)


def _safe_args(args: dict) -> dict:
    return {k: (v[:SAFE_ARG_CHARS] if isinstance(v, str) else v) for k, v in (args or {}).items()}


async def run_tool_loop(
    ctx: RequestContext,
    *,
    agent: str,
    system: str,
    user: str,
    tools: list[BaseTool],
    max_rounds: int,
    require_first_tool: bool = True,
) -> LoopOutput:
    by_name = {t.name: t for t in tools}
    messages: list[BaseMessage] = [SystemMessage(system), HumanMessage(user)]
    out = LoopOutput(text="", rounds=0)

    for rnd in range(1, max_rounds + 1):
        out.rounds = rnd
        decision = ctx.route("worker_tool_loop", agent, _loop_features(ctx, messages, rnd, out))
        choice = "required" if (rnd == 1 and require_first_tool) else "auto"
        ai = await call_llm(ctx, decision, lambda c: c.bind_tools(tools, tool_choice=choice), messages)
        messages.append(ai)
        if not ai.tool_calls:
            out.text = text_of(ai.content)
            return out
        for i, call in enumerate(ai.tool_calls):
            if i >= MAX_CALLS_PER_ROUND or out.tool_calls >= MAX_CALLS_PER_TASK:
                messages.append(ToolMessage("skipped: tool call budget reached; answer from the evidence you have",
                                            tool_call_id=call["id"]))
                continue
            messages.append(await _run_tool(ctx, agent, by_name, call, out))

    # Tool budget spent: one last round with tools disabled so the worker writes up what it has.
    out.rounds += 1
    messages.append(HumanMessage("Tool budget reached. Write your findings now from the evidence above."))
    decision = ctx.route("worker_final", agent, {**_loop_features(ctx, messages, out.rounds, out),
                                                  "reason": "tool round cap reached"})
    ai = await call_llm(ctx, decision, lambda c: c.bind_tools(tools, tool_choice="none"), messages)
    out.text = text_of(ai.content)
    return out


def _loop_features(ctx: RequestContext, messages: list[BaseMessage], rnd: int, out: LoopOutput) -> dict:
    """Observable inputs for routing a tool-loop call (logged in the llm_route event)."""
    return {
        "round": rnd,
        "tool_calls_so_far": out.tool_calls,
        "context_tokens_est": sum(approx_tokens(text_of(m.content)) for m in messages),
        "comparison": comparison_cue(ctx.query),
    }


async def _run_tool(ctx: RequestContext, agent: str, by_name: dict, call: dict, out: LoopOutput) -> ToolMessage:
    name = call["name"]
    out.tool_calls += 1
    ctx.emitter.emit("tool_call", agent=agent, tool=name, args_safe=_safe_args(call.get("args")))
    t0 = time.perf_counter()
    tool = by_name.get(name)
    try:
        if tool is None:
            raise ValueError(f"unknown tool {name!r}")
        # Invoking with the full tool call returns a ToolMessage; content_and_artifact tools put the
        # cited ids in .artifact, which gives the result count for the trace.
        msg = await tool.ainvoke({**call, "type": "tool_call"})
        content = msg.content
        n = len(msg.artifact) if isinstance(getattr(msg, "artifact", None), list) else None
        status = "ok"
    except Exception as e:  # tool failure is reported to the model and the trace, not raised
        content, n, status = f"ERROR: {type(e).__name__}: {e}"[:500], 0, "error"
        out.tool_errors.append(f"{name}: {type(e).__name__}")
    ctx.emitter.emit(
        "tool_result",
        agent=agent,
        tool=name,
        status=status,
        latency_ms=int((time.perf_counter() - t0) * 1000),
        n_results=n,
    )
    return ToolMessage(str(content), tool_call_id=call["id"])
