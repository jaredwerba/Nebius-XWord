"""Evaluation harness: run a solver over a directory of puzzles and report metrics.

Usage (from the repo root):
    PYTHONPATH=src python3 -m eval.run_eval --solver oracle
    PYTHONPATH=src python3 -m eval.run_eval --solver llm --runs 3

Solvers:
    empty      leave the grid blank (floor / harness sanity check)
    oracle     fill from the answer key (ceiling / harness sanity check)
    backtrack  wordlist backtracking, ignores clues (structure-only baseline)
    llm        run the Nebius-XWord agent (needs an API key, see .env.example)

LLM solves are stochastic: use --runs N to repeat each puzzle and report means.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from nebius_xword.grid import Grid, Puzzle, load_puzzle
from nebius_xword.solver import fill_grid

from .metrics import score_grid

DEFAULT_WORDLIST = "/usr/share/dict/words"


def solve_empty(puzzle: Puzzle, run: int, ctx: dict) -> tuple[Grid, dict]:
    return puzzle.make_grid(), {}


def solve_oracle(puzzle: Puzzle, run: int, ctx: dict) -> tuple[Grid, dict]:
    grid = puzzle.make_grid()
    grid.set_rows(puzzle.solution)
    return grid, {}


def solve_backtrack(puzzle: Puzzle, run: int, ctx: dict) -> tuple[Grid, dict]:
    grid = puzzle.make_grid()
    fill_grid(grid, ctx["words"], rng=random.Random(run))
    return grid, {}


def solve_llm(puzzle: Puzzle, run: int, ctx: dict) -> tuple[Grid, dict]:
    result = ctx["agent"].solve(puzzle)
    return result.grid, {"turns": result.turns, "tokens": result.total_tokens}


SOLVERS = {
    "empty": solve_empty,
    "oracle": solve_oracle,
    "backtrack": solve_backtrack,
    "llm": solve_llm,
}


def build_context(args: argparse.Namespace) -> dict:
    ctx: dict = {}
    if args.solver == "backtrack":
        path = Path(args.wordlist)
        if not path.exists():
            raise SystemExit(f"wordlist not found: {path} (pass --wordlist)")
        ctx["words"] = [w for w in path.read_text().split() if w.islower() and w.isalpha()]
    if args.solver == "llm":
        from nebius_xword.agent import CrosswordAgent  # deferred: needs openai + key

        ctx["agent"] = CrosswordAgent(model=args.model)
    return ctx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puzzles", default="data/puzzles", help="directory of puzzle JSON files")
    parser.add_argument("--solver", choices=sorted(SOLVERS), default="llm")
    parser.add_argument("--runs", type=int, default=1, help="repeat runs per puzzle (mean reported)")
    parser.add_argument("--model", default=None, help="override the model (llm solver)")
    parser.add_argument("--wordlist", default=DEFAULT_WORDLIST, help="wordlist (backtrack solver)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    paths = sorted(Path(args.puzzles).glob("*.json"))
    if not paths:
        raise SystemExit(f"no puzzle files found in {args.puzzles}")
    ctx = build_context(args)

    results = []
    for path in paths:
        puzzle = load_puzzle(path)
        if puzzle.solution is None:
            print(f"skipping {puzzle.id}: no answer key")
            continue
        runs = []
        for run in range(args.runs):
            grid, stats = SOLVERS[args.solver](puzzle, run, ctx)
            runs.append({**score_grid(grid, puzzle.solution), **stats})
        n = len(runs)
        row = {
            "puzzle": puzzle.id,
            "runs": n,
            "letter_accuracy": sum(r["letter_accuracy"] for r in runs) / n,
            "word_accuracy": sum(r["word_accuracy"] for r in runs) / n,
            "solved": sum(r["solved"] for r in runs),
        }
        for key in ("turns", "tokens"):
            if any(key in r for r in runs):
                row[key] = sum(r.get(key, 0) for r in runs) / n
        results.append(row)

    if args.json:
        print(json.dumps({"solver": args.solver, "runs": args.runs, "results": results}, indent=2))
        return

    has_stats = any("turns" in r for r in results)
    print(f"solver: {args.solver} | runs per puzzle: {args.runs}\n")
    header = f"{'puzzle':<20} {'letters':>8} {'words':>8} {'solved':>8}"
    if has_stats:
        header += f" {'turns':>7} {'tokens':>9}"
    print(header)
    for row in results:
        line = (
            f"{row['puzzle']:<20} {row['letter_accuracy']:>7.0%} "
            f"{row['word_accuracy']:>8.0%} {row['solved']:>5}/{row['runs']}"
        )
        if has_stats:
            line += f" {row.get('turns', 0):>7.1f} {row.get('tokens', 0):>9.0f}"
        print(line)
    n = len(results)
    if n:
        mean_l = sum(r["letter_accuracy"] for r in results) / n
        mean_w = sum(r["word_accuracy"] for r in results) / n
        solved = sum(r["solved"] for r in results)
        total = sum(r["runs"] for r in results)
        print(f"\n{'mean / total':<20} {mean_l:>7.0%} {mean_w:>8.0%} {solved:>5}/{total}")


if __name__ == "__main__":
    main()
