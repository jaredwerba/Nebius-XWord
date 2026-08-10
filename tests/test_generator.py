"""Offline tests for puzzle generation: grid fill is real, clues are safe."""

import json
import random

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from nebius_xword.generator import (
    TEMPLATES,
    fill_random_template,
    load_wordlist,
    mask_answer,
    to_puzzle,
    write_clues,
)
from nebius_xword.grid import EMPTY, Grid
from nebius_xword.solver import verify_puzzle

WORDS = load_wordlist()


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_every_template_fills_with_real_words(name):
    chosen, grid = fill_random_template(rng=random.Random(0), template=name, words=WORDS)
    assert chosen == name
    assert grid.is_complete()
    vocabulary = set(WORDS)
    for slot_id in grid.slots:
        assert grid.slot_pattern(slot_id) in vocabulary  # every entry is a real word


def test_fill_is_seed_reproducible():
    a = fill_random_template(rng=random.Random(5), words=WORDS)[1].render()
    b = fill_random_template(rng=random.Random(5), words=WORDS)[1].render()
    assert a == b


def test_fill_varies_across_seeds():
    grids = {fill_random_template(rng=random.Random(s), words=WORDS)[1].render() for s in range(5)}
    assert len(grids) > 1


def fake_clue_model(grid, clue_for, title="Test Puzzle", extra=""):
    """A model that always replies with the same clue payload.

    The reply is queued twice so a validation retry has something to read.
    """
    payload = {
        "title": title,
        "clues": {slot_id: clue_for(grid.slot_pattern(slot_id)) for slot_id in grid.slots},
    }
    reply = AIMessage(content=extra + json.dumps(payload))
    return GenericFakeChatModel(messages=iter([reply, reply]))


# Clue text made of non-words, so it can never collide with an answer and
# trigger the leak masking that these tests are not exercising.
def neutral_clue(answer: str) -> str:
    return f"Qqq zzz {len(answer)} xyzzy"


@pytest.fixture
def filled():
    return fill_random_template(rng=random.Random(1), template="5x5-a", words=WORDS)


def test_write_clues_returns_one_clue_per_slot(filled):
    _, grid = filled
    model = fake_clue_model(grid, neutral_clue)
    title, clues = write_clues(model, grid)
    assert title == "Test Puzzle"
    assert set(clues) == set(grid.slots)


def test_clues_wrapped_in_code_fences_are_parsed(filled):
    _, grid = filled
    model = fake_clue_model(grid, neutral_clue, extra="```json\n")
    _, clues = write_clues(model, grid)
    assert len(clues) == len(grid.slots)


def test_leaked_answer_is_masked(filled):
    _, grid = filled
    # Every clue names its own answer — the generator must blank them out.
    model = fake_clue_model(grid, lambda answer: f"The answer is {answer.title()} here")
    _, clues = write_clues(model, grid)
    for slot_id, clue in clues.items():
        assert grid.slot_pattern(slot_id).lower() not in clue.lower()
        assert "___" in clue


def test_mask_answer_leaves_clean_clues_alone():
    assert mask_answer("Feline pet", "CAT") == "Feline pet"
    assert mask_answer("A cat naps", "CAT") == "A ___ naps"
    assert mask_answer("Cats and dogs", "CAT") == "___ and dogs"  # also catches plurals


def test_missing_clue_raises(filled):
    _, grid = filled
    empty = GenericFakeChatModel(
        messages=iter([AIMessage(content='{"title": "x", "clues": {}}')] * 2)
    )
    with pytest.raises(RuntimeError, match="never clued"):
        write_clues(empty, grid)


def test_generated_puzzle_is_valid_and_solvable_blind(filled):
    name, grid = filled
    model = fake_clue_model(grid, neutral_clue)
    title, clues = write_clues(model, grid)
    puzzle = to_puzzle(name, grid, title, clues)

    assert verify_puzzle(puzzle) == []  # numbering, clue coverage, key consistency
    assert puzzle.solution == grid.render().split("\n")
    # The solver's view starts blank and carries no answers.
    fresh = puzzle.make_grid()
    assert all(EMPTY in fresh.slot_pattern(slot_id) for slot_id in fresh.slots)
    assert all(slot.clue for slot in fresh.slots.values())


def test_template_grids_are_wellformed():
    for name, rows in TEMPLATES.items():
        grid = Grid(rows)
        assert grid.slots, f"{name} has no slots"
        assert all(slot.length >= 3 for slot in grid.slots.values()), name
