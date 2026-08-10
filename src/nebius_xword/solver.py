"""Deterministic (non-LLM) solving utilities.

Currently: pattern matching against a wordlist and puzzle-file validation.
Planned: a backtracking constraint solver to serve as a non-LLM baseline and
as a repair step for near-complete LLM fills.
"""

from __future__ import annotations

from typing import Iterable

from .grid import BLOCK, EMPTY, Puzzle


def matches(pattern: str, word: str) -> bool:
    """True if ``word`` fits ``pattern`` ('.' matches any letter)."""
    word = word.upper()
    return len(word) == len(pattern) and all(
        p == EMPTY or p == w for p, w in zip(pattern.upper(), word)
    )


def candidates(pattern: str, words: Iterable[str]) -> list[str]:
    """All words from a wordlist that fit a slot pattern."""
    return [w.upper() for w in words if matches(pattern, w)]


def verify_puzzle(puzzle: Puzzle) -> list[str]:
    """Sanity-check a puzzle file. Returns a list of problems (empty = valid)."""
    problems: list[str] = []
    grid = puzzle.make_grid()
    for slot in grid.slots.values():
        if not slot.clue:
            problems.append(f"{slot.id} has no clue")
    if puzzle.solution is None:
        return problems
    sol = puzzle.solution
    if len(sol) != grid.rows or any(len(row) != grid.cols for row in sol):
        return problems + ["solution dimensions do not match grid"]
    for r in range(grid.rows):
        for c in range(grid.cols):
            tpl_block = grid.template[r][c] == BLOCK
            sol_ch = sol[r][c]
            if tpl_block != (sol_ch == BLOCK):
                problems.append(f"block mismatch at ({r},{c})")
            elif not tpl_block and not sol_ch.isalpha():
                problems.append(f"solution cell ({r},{c}) is not a letter: {sol_ch!r}")
    return problems
