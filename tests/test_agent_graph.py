"""Offline tests for the LangGraph solver loop using a scripted fake model.

GenericFakeChatModel returns each scripted AIMessage verbatim and raises
StopIteration if the graph asks for more turns than scripted — a deliberate
guard against over-looping.
"""

from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from nebius_xword.agent import CrosswordAgent
from nebius_xword.graph import build_solver_graph
from nebius_xword.grid import load_puzzle
from nebius_xword.tools import ToolExecutor

PUZZLES = Path(__file__).parents[1] / "data" / "puzzles"


def tool_call(name, args, call_id):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def ai_turn(calls, usage=None, content=""):
    return AIMessage(content=content, tool_calls=calls, usage_metadata=usage)


USAGE = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}

SOLVE_3X3_SCRIPT = [
    ai_turn(
        [
            tool_call("fill_slot", {"slot": "1A", "word": "CAT"}, "c1"),
            tool_call("fill_slot", {"slot": "4A", "word": "ARE"}, "c2"),
            tool_call("fill_slot", {"slot": "5A", "word": "TEN"}, "c3"),
        ],
        usage=USAGE,
    ),
    ai_turn(
        [
            tool_call("fill_slot", {"slot": "1D", "word": "CAT"}, "c4"),
            tool_call("fill_slot", {"slot": "2D", "word": "ARE"}, "c5"),
            tool_call("fill_slot", {"slot": "3D", "word": "TEN"}, "c6"),
            tool_call("submit", {}, "c7"),
        ],
        usage=USAGE,
    ),
]


@pytest.fixture
def puzzle():
    return load_puzzle(PUZZLES / "example_3x3.json")


def run_graph(puzzle, script, max_turns=24):
    executor = ToolExecutor(puzzle)
    fake = GenericFakeChatModel(messages=iter(script))
    graph = build_solver_graph(fake, executor, max_turns=max_turns)
    final = graph.invoke(
        {"messages": [], "turns": 0, "nudges": 0},
        config={"recursion_limit": 2 * max_turns + 12},
    )
    return executor, final


def test_submit_stops_graph_and_fills_grid(puzzle):
    executor, final = run_graph(puzzle, SOLVE_3X3_SCRIPT)
    assert executor.submitted is True
    assert executor.grid.render() == "CAT\nARE\nTEN"
    tool_messages = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_messages] == ["c1", "c2", "c3", "c4", "c5", "c6", "c7"]
    assert final["turns"] == 2


def test_agent_end_to_end_with_fake_model(puzzle):
    fake = GenericFakeChatModel(messages=iter(SOLVE_3X3_SCRIPT))
    agent = CrosswordAgent(chat_model=fake)
    result = agent.solve(puzzle)
    assert result.submitted is True
    assert result.turns == 2
    assert result.prompt_tokens == 200
    assert result.completion_tokens == 40
    assert result.total_tokens == 240
    assert result.grid.is_complete()


def test_prose_only_model_is_nudged_then_run_ends(puzzle):
    # Four prose replies: the initial one plus one per allowed nudge (3).
    script = [AIMessage(content="I give up.") for _ in range(4)]
    fake = GenericFakeChatModel(messages=iter(script))
    result = CrosswordAgent(chat_model=fake).solve(puzzle)
    assert result.submitted is False
    assert result.turns == 4  # a model that never calls tools still terminates
    assert not result.grid.is_complete()


def test_nudge_recovers_a_prose_start(puzzle):
    script = [AIMessage(content="Let me think about this puzzle first...")] + SOLVE_3X3_SCRIPT
    fake = GenericFakeChatModel(messages=iter(script))
    result = CrosswordAgent(chat_model=fake).solve(puzzle)
    assert result.submitted is True
    assert result.grid.is_complete()
    assert result.turns == 3  # prose turn + the two scripted solving turns


def test_max_turns_cap(puzzle):
    # 4 scripted get_state turns but max_turns=3: the graph must stop at 3
    # (the fake raises StopIteration if a 4th call were made).
    script = [ai_turn([tool_call("get_state", {}, f"g{i}")]) for i in range(4)]
    executor, final = run_graph(puzzle, script, max_turns=3)
    assert final["turns"] == 3
    assert executor.submitted is False
    # the 3rd turn's tool call still executed
    assert sum(isinstance(m, ToolMessage) for m in final["messages"]) == 3


def test_missing_usage_metadata_tolerated(puzzle):
    script = [
        ai_turn([tool_call("get_state", {}, "g1")], usage=None),
        ai_turn([tool_call("submit", {}, "s1")], usage=USAGE),
    ]
    fake = GenericFakeChatModel(messages=iter(script))
    result = CrosswordAgent(chat_model=fake).solve(puzzle)
    assert result.prompt_tokens == 100
    assert result.completion_tokens == 20


def test_tool_error_keeps_loop_alive(puzzle):
    script = [
        ai_turn([tool_call("fill_slot", {"slot": "1A", "word": "TOOLONG"}, "e1")]),
        ai_turn([tool_call("submit", {}, "s1")]),
    ]
    executor, final = run_graph(puzzle, script)
    error_msg = next(m for m in final["messages"] if isinstance(m, ToolMessage))
    assert "error" in error_msg.content
    assert executor.submitted is True  # loop survived the error and continued


def test_window_keeps_head_and_recent_tail():
    from langchain_core.messages import HumanMessage, SystemMessage

    from nebius_xword.graph import window_messages

    head = [SystemMessage(content="sys"), HumanMessage(content="grid")]
    body = []
    for i in range(10):
        body.append(ai_turn([tool_call("get_state", {}, f"c{i}")]))
        body.append(ToolMessage(content="{}", tool_call_id=f"c{i}", name="get_state"))
    msgs = head + body

    out = window_messages(msgs, keep_last=5)
    assert out[:2] == head  # system prompt and initial grid always survive
    assert len(out) <= 2 + 5
    assert not isinstance(out[2], ToolMessage)  # no orphaned tool result after cut

    assert window_messages(msgs, keep_last=100) == msgs  # short histories untouched
    assert window_messages(msgs, keep_last=0) == msgs  # zero disables the window


def test_stream_yields_progress_events_and_result(puzzle):
    fake = GenericFakeChatModel(messages=iter(SOLVE_3X3_SCRIPT))
    events = list(CrosswordAgent(chat_model=fake).stream(puzzle))
    kinds = [e["event"] for e in events]
    assert kinds[0] == "start"
    assert kinds[-1] == "result"
    assert kinds.count("llm") == 2
    assert kinds.count("tool_call") == 7
    assert kinds.count("tool_result") == 7
    first_call = next(e for e in events if e["event"] == "tool_call")
    assert first_call["tool"] == "fill_slot" and first_call["args"]["slot"] == "1A"
    result = events[-1]["result"]
    assert result.submitted is True and result.turns == 2
    assert result.total_tokens == 240


def test_parallel_calls_with_submit_all_execute(puzzle):
    script = [
        ai_turn(
            [
                tool_call("fill_slot", {"slot": "1A", "word": "CAT"}, "p1"),
                tool_call("submit", {}, "p2"),
            ]
        ),
    ]
    executor, final = run_graph(puzzle, script)
    assert executor.submitted is True
    assert executor.grid.slot_pattern("1A") == "CAT"  # fill before submit both ran
    assert [m.tool_call_id for m in final["messages"] if isinstance(m, ToolMessage)] == ["p1", "p2"]
