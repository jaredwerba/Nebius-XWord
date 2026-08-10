"""Evaluation harness: run a solver over a directory of puzzles and report metrics.

Usage (from the repo root):
    PYTHONPATH=src python3 -m eval.run_eval --solver oracle
    PYTHONPATH=src python3 -m eval.run_eval --solver llm --puzzles data/puzzles

Solvers:
    empty   leave the grid blank (floor / harness sanity check)
    oracle  fill from the answer key (ceiling / harness sanity check)
    llm     run the Nebius-XWord agent (needs NEBIUS_API_KEY in .env)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nebius_xword.grid import Grid, Puzzle, load_puzzle

from .metrics import score_grid


def solve_empty(puzzle: Puzzle) -> Grid:
    return puzzle.make_grid()


def solve_oracle(puzzle: Puzzle) -> Grid:
    grid = puzzle.make_grid()
    grid.set_rows(puzzle.solution)
    return grid


def solve_llm(puzzle: Puzzle) -> Grid:
    from nebius_xword.agent import CrosswordAgent  # deferred: needs openai + API key

    return CrosswordAgent().solve(puzzle)


SOLVERS = {"empty": solve_empty, "oracle": solve_oracle, "llm": solve_llm}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puzzles", default="data/puzzles", help="directory of puzzle JSON files")
    parser.add_argument("--solver", choices=sorted(SOLVERS), default="llm")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    paths = sorted(Path(args.puzzles).glob("*.json"))
    if not paths:
        raise SystemExit(f"no puzzle files found in {args.puzzles}")

    results = []
    for path in paths:
        puzzle = load_puzzle(path)
        if puzzle.solution is None:
            print(f"skipping {puzzle.id}: no answer key")
            continue
        grid = SOLVERS[args.solver](puzzle)
        scores = score_grid(grid, puzzle.solution)
        results.append({"puzzle": puzzle.id, **scores})

    if args.json:
        print(json.dumps({"solver": args.solver, "results": results}, indent=2))
        return

    print(f"solver: {args.solver}\n")
    print(f"{'puzzle':<24} {'letters':>8} {'words':>8} {'solved':>7}")
    for row in results:
        print(
            f"{row['puzzle']:<24} {row['letter_accuracy']:>7.0%} "
            f"{row['word_accuracy']:>8.0%} {str(row['solved']):>7}"
        )
    n = len(results)
    if n:
        mean_l = sum(r["letter_accuracy"] for r in results) / n
        mean_w = sum(r["word_accuracy"] for r in results) / n
        solved = sum(r["solved"] for r in results)
        print(f"\n{'mean / total':<24} {mean_l:>7.0%} {mean_w:>8.0%} {solved:>4}/{n}")


if __name__ == "__main__":
    main()
