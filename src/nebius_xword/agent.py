"""LLM crossword agent: a LangGraph tool loop over an OpenAI-compatible endpoint.

Two supported backends, both via ``langchain-openai``'s ChatOpenAI:

- **Vercel AI Gateway** (default): set ``LLM_API_KEY`` (or run on Vercel, where
  the OIDC token is used automatically). Models use gateway slugs like
  ``deepseek/deepseek-v4-flash-0731``.
- **Nebius AI Studio** (direct): set ``NEBIUS_API_KEY``; models use Nebius
  slugs like ``meta-llama/Meta-Llama-3.1-70B-Instruct``.

Explicit constructor arguments always win over environment variables. The
control flow lives in :mod:`nebius_xword.graph`; this module resolves config,
builds the model, and assembles the SolveResult.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError

from .graph import build_solver_graph
from .grid import Grid, Puzzle
from .tools import TOOL_SCHEMAS, ToolExecutor

GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"
GATEWAY_DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"
NEBIUS_BASE_URL = "https://api.studio.nebius.com/v1/"
NEBIUS_DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"

SYSTEM_PROMPT = """\
You are Nebius-XWord, an expert crossword solver.

The full grid and clues are in the first message, so do not open with
get_state — start filling immediately. Every turn costs time: fill as many
slots as you can in each one, using parallel tool calls.

Strategy:
1. Fill the answers you are most confident about first, then the
   most-constrained slots: the ones whose patterns already contain crossing
   letters. Use those letters to narrow the answer.
2. Each fill_slot result reports the resulting pattern, so you can track the
   grid yourself. Call get_state only when you have lost track.
3. If fill_slot reports conflicts, one of the crossing answers is wrong.
   Use clear_slot to back out and try alternatives. Never re-try an answer
   that was already rejected or cleared — enumerate different candidates.
4. Clue conventions: a clue ending in '?' is wordplay — read it literally or
   punnily. Multi-word answers are written with no spaces or punctuation.
   Abbreviated clues want abbreviated answers.
5. Before submitting, re-check any slot whose letters were mostly forced by
   crossings: confirm the word actually answers its own clue. If it does not,
   clear and fix it first.
6. When every slot is filled and consistent, call submit.

Answers contain letters only (no spaces or punctuation)."""


def resolve_llm_config(
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> tuple[str, str | None, str]:
    """Resolve (model, api_key, base_url) from args and environment."""
    load_dotenv()
    api_key = (
        api_key
        or os.getenv("LLM_API_KEY")
        or os.getenv("NEBIUS_API_KEY")
        or os.getenv("VERCEL_OIDC_TOKEN")  # auto-auth for AI Gateway on Vercel
    )
    if base_url is None:
        base_url = os.getenv("LLM_BASE_URL") or (
            NEBIUS_BASE_URL
            if os.getenv("NEBIUS_API_KEY") and not os.getenv("LLM_API_KEY")
            else GATEWAY_BASE_URL
        )
    if model is None:
        model = os.getenv("LLM_MODEL") or os.getenv("NEBIUS_MODEL") or (
            NEBIUS_DEFAULT_MODEL if "nebius" in base_url else GATEWAY_DEFAULT_MODEL
        )
    return model, api_key, base_url


def build_chat_model(
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
):
    """Build an unbound chat model against the resolved endpoint."""
    from langchain_openai import ChatOpenAI  # deferred: heavy import

    name, key, url = resolve_llm_config(model, api_key, base_url)
    return ChatOpenAI(
        model=name,
        api_key=key,
        base_url=url,
        use_responses_api=False,  # pin the Chat Completions surface
    )


@dataclass
class SolveResult:
    grid: Grid
    turns: int
    prompt_tokens: int
    completion_tokens: int
    submitted: bool
    model: str

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class CrosswordAgent:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_turns: int = 24,
        verbose: bool = False,
        *,
        chat_model=None,
    ):
        """``chat_model`` injects a pre-built LangChain chat model (tests)."""
        self.model, _, _ = resolve_llm_config(model, api_key, base_url)
        if chat_model is not None:
            self._model = chat_model
        else:
            self._model = build_chat_model(model, api_key, base_url).bind_tools(TOOL_SCHEMAS)
        self.max_turns = max_turns
        self.verbose = verbose

    def solve(self, puzzle: Puzzle) -> SolveResult:
        """Run the graph until the model submits, stops, or turns run out."""
        result = None
        for event in self.stream(puzzle):
            if event["event"] == "result":
                result = event["result"]
        return result

    def stream(self, puzzle: Puzzle):
        """Yield progress events while solving; the last event carries the result.

        Events: {"event": "start"}, {"event": "llm", "turn"}, {"event":
        "tool_call", "turn", "tool", "args"}, {"event": "tool_result", "tool",
        "result"}, and finally {"event": "result", "result": SolveResult}.
        """
        executor = ToolExecutor(puzzle)
        graph = build_solver_graph(
            self._model, executor, max_turns=self.max_turns, verbose=self.verbose
        )
        init = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(
                    content="Solve this puzzle:\n" + executor.execute("get_state", {})
                ),
            ],
            "turns": 0,
        }
        yield {"event": "start", "model": self.model, "puzzle": puzzle.id}

        turns = 0
        ai_messages: list[AIMessage] = []
        try:
            for update in graph.stream(
                init,
                config={"recursion_limit": 2 * self.max_turns + 4},
                stream_mode="updates",
            ):
                if "agent" in update:
                    turns = update["agent"]["turns"]
                    message = update["agent"]["messages"][-1]
                    ai_messages.append(message)
                    yield {"event": "llm", "turn": turns}
                    for call in message.tool_calls:
                        yield {
                            "event": "tool_call",
                            "turn": turns,
                            "tool": call["name"],
                            "args": call["args"],
                        }
                elif "tools" in update:
                    for msg in update["tools"]["messages"]:
                        yield {"event": "tool_result", "tool": msg.name, "result": msg.content}
        except GraphRecursionError:  # backstop; the turns counter normally stops first
            turns = self.max_turns

        prompt_tokens = completion_tokens = 0
        for message in ai_messages:
            usage = getattr(message, "usage_metadata", None)
            if usage:
                prompt_tokens += usage.get("input_tokens", 0) or 0
                completion_tokens += usage.get("output_tokens", 0) or 0

        yield {
            "event": "result",
            "result": SolveResult(
                grid=executor.grid,
                turns=turns,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                submitted=executor.submitted,
                model=self.model,
            ),
        }
