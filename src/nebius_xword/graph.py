"""LangGraph solver graph: an agent node and a tools node over a ToolExecutor.

The graph owns control flow only. The Grid stays the source of truth inside
the ToolExecutor (closure), and the chat model is injected pre-bound to
TOOL_SCHEMAS — fake chat models used in tests don't implement bind_tools.

    START -> agent
    agent -> tools   (last AIMessage has tool_calls)  |  END (no tool calls)
    tools -> agent   (keep going)  |  END (submit ran, or max_turns reached)
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from .tools import ToolExecutor


class AgentState(MessagesState):
    turns: int  # number of LLM calls made so far


def window_messages(messages: list, keep_last: int) -> list:
    """Bound what the model re-reads each turn.

    Without this, every turn resends the whole history, so token cost grows
    with the square of the turn count — measured: 1.09M tokens for one
    40-turn solve. The window keeps the head (system prompt + the initial
    grid-and-clues message) plus the most recent ``keep_last`` messages, so
    cost grows linearly and long solves stay affordable.

    The cut must not orphan tool results: an OpenAI-style API rejects a tool
    message whose assistant call is missing, so leading ToolMessages after
    the cut are dropped. The model can re-sync any state that scrolled out
    of the window by calling get_state.
    """
    head, body = messages[:2], messages[2:]
    if keep_last <= 0 or len(body) <= keep_last:
        return messages
    tail = body[-keep_last:]
    while tail and isinstance(tail[0], ToolMessage):
        tail = tail[1:]
    return head + tail


def build_solver_graph(
    model: BaseChatModel,
    executor: ToolExecutor,
    *,
    max_turns: int,
    history_window: int = 48,
    verbose: bool = False,
):
    """Compile a solver graph for one solve (compile is milliseconds)."""

    def agent_node(state: AgentState) -> dict:
        message = model.invoke(window_messages(state["messages"], history_window))
        return {"messages": [message], "turns": state["turns"] + 1}

    def tools_node(state: AgentState) -> dict:
        results = []
        for call in state["messages"][-1].tool_calls:
            result = executor.execute(call["name"], call["args"])
            if verbose:
                print(f"[turn {state['turns']}] {call['name']}({call['args']}) -> {result}")
            results.append(
                ToolMessage(content=result, tool_call_id=call["id"], name=call["name"])
            )
        return {"messages": results}

    def after_agent(state: AgentState) -> str:
        return "tools" if state["messages"][-1].tool_calls else END

    def after_tools(state: AgentState) -> str:
        if executor.submitted or state["turns"] >= max_turns:
            return END
        return "agent"

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", after_agent, ["tools", END])
    builder.add_conditional_edges("tools", after_tools, ["agent", END])
    return builder.compile()
