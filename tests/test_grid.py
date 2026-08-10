from pathlib import Path

import pytest

from nebius_xword.grid import EMPTY, Grid, load_puzzle
from nebius_xword.solver import verify_puzzle

PUZZLES = Path(__file__).parents[1] / "data" / "puzzles"


@pytest.fixture
def mini():
    return load_puzzle(PUZZLES / "example_mini_5x5.json")


def test_slot_numbering(mini):
    grid = mini.make_grid()
    across = [s.id for s in grid.slots.values() if s.direction == "across"]
    down = [s.id for s in grid.slots.values() if s.direction == "down"]
    assert across == ["1A", "4A", "5A", "6A", "7A"]
    assert down == ["1D", "2D", "3D", "4D", "5D"]


def test_slot_lengths(mini):
    grid = mini.make_grid()
    lengths = {sid: slot.length for sid, slot in grid.slots.items()}
    assert lengths == {
        "1A": 3, "4A": 4, "5A": 5, "6A": 4, "7A": 3,
        "1D": 5, "2D": 4, "3D": 3, "4D": 4, "5D": 3,
    }


def test_fill_reports_conflicts(mini):
    grid = mini.make_grid()
    assert grid.fill_slot("1A", "pit") == []
    assert grid.slot_pattern("1D") == "P...."
    conflicts = grid.fill_slot("1D", "QUOTA")
    assert len(conflicts) == 1 and grid.slot_pattern("1D") == "P...."  # unchanged
    assert grid.fill_slot("1D", "PILOT") == []


def test_fill_validation(mini):
    grid = mini.make_grid()
    with pytest.raises(ValueError):
        grid.fill_slot("1A", "TOOLONG")
    with pytest.raises(ValueError):
        grid.fill_slot("1A", "P1T")
    with pytest.raises(KeyError):
        grid.fill_slot("99A", "XYZ")


def test_clear_slot(mini):
    grid = mini.make_grid()
    grid.fill_slot("1A", "PIT")
    grid.clear_slot("1A")
    assert grid.slot_pattern("1A") == EMPTY * 3


def test_solution_completes_grid(mini):
    grid = mini.make_grid()
    grid.set_rows(mini.solution)
    assert grid.is_complete()


@pytest.mark.parametrize("name", ["example_3x3.json", "example_mini_5x5.json"])
def test_example_puzzles_are_valid(name):
    assert verify_puzzle(load_puzzle(PUZZLES / name)) == []


def test_rejects_bad_template():
    with pytest.raises(ValueError):
        Grid(["..", "..."])  # ragged
    with pytest.raises(ValueError):
        Grid([".x", ".."])  # bad character
