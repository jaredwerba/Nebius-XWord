"""Tests for the external-puzzle importer. All data here is synthetic."""

import pytest

from nebius_xword.external import audit_fill, check_import, compare_fills, puzzle_from_scrape
from nebius_xword.grid import Grid

# A 3x3 open square numbers as: 1A/4A/5A across, 1D/2D/3D down.
SCRAPE = {
    "title": "Test 1",
    "grid": ["...", "...", "..."],
    "across": [["1", "clue a"], ["4", "clue b"], ["5", "clue c"]],
    "down": [["1", "clue d"], ["2", "clue e"], ["3", "clue f"]],
}


def test_scrape_imports_when_numbering_agrees():
    puzzle = puzzle_from_scrape(SCRAPE, source="test")
    assert puzzle.id == "test-test-1"
    assert puzzle.solution is None  # imported puzzles never carry a key
    grid = puzzle.make_grid()
    assert len(grid.slots) == 6
    assert all(slot.clue for slot in grid.slots.values())


def test_scrape_with_wrong_numbers_is_rejected():
    bad = {**SCRAPE, "across": [["1", "a"], ["3", "b"], ["5", "c"]]}  # 3A does not exist
    with pytest.raises(ValueError, match="inconsistent"):
        puzzle_from_scrape(bad)


def test_check_import_reports_both_directions():
    puzzle = puzzle_from_scrape(SCRAPE)
    puzzle.clues["across"].pop("4")
    puzzle.clues["down"]["9"] = "phantom"
    problems = check_import(puzzle)
    assert "4A has no clue" in problems
    assert "clue 9D matches no slot" in problems


def test_audit_fill_flags_invented_words():
    grid = Grid(["...", "...", "..."])
    grid.set_rows(["CAT", "ARE", "TEN"])
    words = {"CAT", "ARE", "TEN"}  # columns CAT/ARE/TEN also in set
    audit = audit_fill(grid, words)
    assert audit["complete"] is True
    assert audit["in_vocabulary"] == 1.0
    audit2 = audit_fill(grid, {"CAT"})
    assert audit2["in_vocabulary"] < 1.0
    assert "TEN" in audit2["unknown_words"]


def test_compare_fills_reports_disagreements():
    a, b = Grid(["...", "...", "..."]), Grid(["...", "...", "..."])
    a.set_rows(["CAT", "ARE", "TEN"])
    b.set_rows(["CAT", "ARE", "TON"])
    cmp = compare_fills(a, b)
    assert cmp["slots"] == 6
    assert 0 < cmp["agreement"] < 1
    assert "5A" in cmp["disagreements"]
