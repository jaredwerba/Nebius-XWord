from pathlib import Path

import pytest

from eval.metrics import score_grid
from nebius_xword.grid import load_puzzle

PUZZLES = Path(__file__).parents[1] / "data" / "puzzles"


@pytest.fixture
def mini():
    return load_puzzle(PUZZLES / "example_mini_5x5.json")


def test_oracle_scores_perfect(mini):
    grid = mini.make_grid()
    grid.set_rows(mini.solution)
    scores = score_grid(grid, mini.solution)
    assert scores == {"letter_accuracy": 1.0, "word_accuracy": 1.0, "solved": True}


def test_empty_scores_zero(mini):
    scores = score_grid(mini.make_grid(), mini.solution)
    assert scores == {"letter_accuracy": 0.0, "word_accuracy": 0.0, "solved": False}


def test_partial_fill(mini):
    grid = mini.make_grid()
    grid.fill_slot("1A", "PIT")  # 3 of 19 open cells, 1 of 10 words
    scores = score_grid(grid, mini.solution)
    assert scores["letter_accuracy"] == pytest.approx(3 / 19)
    assert scores["word_accuracy"] == pytest.approx(1 / 10)
    assert scores["solved"] is False
