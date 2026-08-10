"""LLM crossword agent: a tool-use loop over an OpenAI-compatible endpoint.

Two supported backends, both via the standard ``openai`` client:

- **Vercel AI Gateway** (default): set ``LLM_API_KEY`` (or run on Vercel, where
  the OIDC token is used automatically). Models use gateway slugs like
  ``openai/gpt-4o-mini``.
- **Nebius AI Studio** (direct): set ``NEBIUS_API_KEY``; models use Nebius
  slugs like ``meta-llama/Meta-Llama-3.1-70B-Instruct``.

Explicit constructor arguments always win over environment variables.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

from .grid import Grid, Puzzle
from .tools import TOOL_SCHEMAS, ToolExecutor

GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"
GATEWAY_DEFAULT_MODEL = "openai/gpt-4o-mini"
NEBIUS_BASE_URL = "https://api.studio.nebius.com/v1/"
NEBIUS_DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"

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
    ):
        self.model, key, url = resolve_llm_config(model, api_key, base_url)
        self.client = OpenAI(api_key=key, base_url=url)
        self.max_turns = max_turns
        self.verbose = verbose

    def solve(self, puzzle: Puzzle) -> SolveResult:
        """Run the tool loop until the model submits or turns run out."""
        executor = ToolExecutor(puzzle)
        messages: list = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Solve this puzzle:\n" + executor.execute("get_state", {})},
        ]
        turns = prompt_tokens = completion_tokens = 0
        for turn in range(self.max_turns):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_SCHEMAS,
            )
            turns += 1
            if response.usage is not None:
                prompt_tokens += response.usage.prompt_tokens or 0
                completion_tokens += response.usage.completion_tokens or 0
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
        return SolveResult(
            grid=executor.grid,
            turns=turns,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            submitted=executor.submitted,
            model=self.model,
        )
