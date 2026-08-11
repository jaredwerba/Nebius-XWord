"""Solve one puzzle with the LLM agent and print the result.

Usage (from the repo root, with a key in .env — see .env.example):
    python scripts/solve.py data/puzzles/example_mini_5x5.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Running a script puts scripts/ on the path, not the repo root, so `eval` and
# an uninstalled `nebius_xword` would both be invisible. Add the root itself.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from nebius_xword.agent import CrosswordAgent  # noqa: E402
from nebius_xword.grid import load_puzzle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("puzzle", help="path to a puzzle JSON file")
    parser.add_argument("--model", default=None, help="override the default model")
    parser.add_argument("--quiet", action="store_true", help="hide the tool-call trace")
    args = parser.parse_args()

    puzzle = load_puzzle(args.puzzle)
    agent = CrosswordAgent(model=args.model, verbose=not args.quiet)
    result = agent.solve(puzzle)

    print(f"\n{puzzle.title}\n{result.grid.render()}")
    print(
        f"model: {result.model} | turns: {result.turns} | "
        f"tokens: {result.total_tokens} | submitted: {result.submitted}"
    )
    if puzzle.solution is not None:
        from eval.metrics import score_grid

        print(f"score: {score_grid(result.grid, puzzle.solution)}")


if __name__ == "__main__":
    main()
