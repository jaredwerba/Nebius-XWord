"""LLM crossword agent: a tool-use loop against Nebius AI Studio.

Nebius AI Studio exposes an OpenAI-compatible API, so the standard ``openai``
client works with a swapped base URL. Configure via ``.env`` (see
``.env.example``) or constructor arguments.
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from .grid import Grid, Puzzle
from .tools import TOOL_SCHEMAS, ToolExecutor

DEFAULT_BASE_URL = "https://api.studio.nebius.com/v1/"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"

SYSTEM_PROMPT = """\
You are Nebius-XWord, an expert crossword solver.

Strategy:
1. Call get_state to see the grid, clues, and current fill patterns.
2. Fill the answers you are most confident about first.
3. Use crossing letters (the '.' patterns) to constrain uncertain answers.
4. If fill_slot reports conflicts, reconsider — one of the crossing answers
   is wrong. Use clear_slot to back out and try alternatives.
5. When every slot is filled and consistent, call submit.

Answers contain letters only (no spaces or punctuation)."""


class CrosswordAgent:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_turns: int = 24,
        verbose: bool = False,
    ):
        load_dotenv()
        self.model = model or os.getenv("NEBIUS_MODEL", DEFAULT_MODEL)
        self.client = OpenAI(
            api_key=api_key or os.getenv("NEBIUS_API_KEY"),
            base_url=base_url or os.getenv("NEBIUS_BASE_URL", DEFAULT_BASE_URL),
        )
        self.max_turns = max_turns
        self.verbose = verbose

    def solve(self, puzzle: Puzzle) -> Grid:
        """Run the tool loop until the model submits or turns run out."""
        executor = ToolExecutor(puzzle)
        messages: list = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Solve this puzzle:\n" + executor.execute("get_state", {})},
        ]
        for turn in range(self.max_turns):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_SCHEMAS,
            )
            message = response.choices[0].message
            messages.append(message)
            if not message.tool_calls:
                break  # model stopped calling tools; take the grid as-is
            for call in message.tool_calls:
                args = json.loads(call.function.arguments or "{}")
                result = executor.execute(call.function.name, args)
                if self.verbose:
                    print(f"[turn {turn}] {call.function.name}({args}) -> {result}")
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
            if executor.submitted:
                break
        return executor.grid
