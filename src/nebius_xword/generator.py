"""Generate fresh crossword puzzles: deterministic grid fill, LLM-written clues.

The split is deliberate. A backtracking search over a shipped wordlist builds
the answer grid, so every entry is a real word and every crossing is
consistent by construction — the model cannot invent a broken puzzle. The
model then does the part it is good at: writing a clue for each answer, and
naming the puzzle.

The solver never sees the answers. It receives the block layout and the clues
only; the answer key is kept for scoring after the solve.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from .grid import Grid, Puzzle
from .solver import fill_grid

WORDLIST_PATH = Path(__file__).resolve().parents[2] / "data" / "wordlist.txt"

# Templates that this wordlist can fill quickly and reliably (verified by
# tests/test_generator.py). '#' is a block, '.' an open cell.
TEMPLATES: dict[str, list[str]] = {
    "5x5-a": ["##...", "#....", ".....", "....#", "...##"],
    "5x5-b": ["...##", "....#", ".....", "#....", "##..."],
    "5x5-c": ["...#.", "...#.", ".....", ".#...", ".#..."],
    "5x5-d": ["....#", "...#.", ".....", ".#...", "#...."],
    "7x7-pinwheel": [
        "...#...", "...#...", ".......", "##...##", ".......", "...#...", "...#...",
    ],
}

CLUE_PROMPT = """\
You are a crossword editor. Write one clue for each answer below.

Rules:
- Never write the answer, or any word sharing its root, inside its own clue.
- Keep each clue under 60 characters. No answer lengths, no numbering.
- Vary the styles: straight definitions, fill-in-the-blank with ___, and
  a little wordplay (mark a wordplay clue with a trailing question mark).
- Also invent a short, playful title for the whole puzzle (under 40 characters).

Answers:
{answers}

Reply with only a JSON object, no prose and no code fences:
{{"title": "...", "clues": {{{example}}}}}"""


def load_wordlist(path: str | Path | None = None) -> list[str]:
    return Path(path or WORDLIST_PATH).read_text().split()


def fill_random_template(
    rng: random.Random | None = None,
    template: str | None = None,
    words: list[str] | None = None,
    attempts: int = 12,
) -> tuple[str, Grid]:
    """Build a filled answer grid. Returns (template name, filled Grid)."""
    rng = rng or random.Random()
    words = words or load_wordlist()
    names = [template] if template else list(TEMPLATES)
    for attempt in range(attempts):
        name = rng.choice(names)
        grid = Grid(TEMPLATES[name])
        if fill_grid(grid, words, rng=rng, max_steps=50_000):
            return name, grid
    raise RuntimeError(f"could not fill any template in {attempts} attempts")


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of a model reply that may carry fences or prose."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start : end + 1])


def mask_answer(clue: str, answer: str) -> str:
    """Blank out the answer if the clue gives it away, keeping the clue usable."""
    return re.sub(rf"\b{re.escape(answer)}\w*\b", "___", clue, flags=re.IGNORECASE)


def _validate(clues: dict, answers: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Return (cleaned clues, problems). Leaked answers are masked, not dropped."""
    cleaned: dict[str, str] = {}
    problems: list[str] = []
    for slot_id, answer in answers.items():
        clue = str(clues.get(slot_id, "")).strip()
        if not clue:
            problems.append(f"{slot_id} has no clue")
            continue
        masked = mask_answer(clue, answer)
        if masked != clue:
            problems.append(f"{slot_id} clue contained the answer")
        cleaned[slot_id] = masked
    return cleaned, problems


def write_clues(model, grid: Grid, retries: int = 1) -> tuple[str, dict[str, str]]:
    """Ask the model for a title and one clue per slot. Returns (title, clues)."""
    answers = {slot_id: grid.slot_pattern(slot_id) for slot_id in grid.slots}
    listing = "\n".join(f"{slot_id}: {answer}" for slot_id, answer in answers.items())
    example = ", ".join(f'"{slot_id}": "..."' for slot_id in list(answers)[:2])
    prompt = CLUE_PROMPT.format(answers=listing, example=example)

    title, cleaned, problems = "", {}, []
    for attempt in range(retries + 1):
        message = prompt if attempt == 0 else (
            f"{prompt}\n\nYour last reply had these problems; fix them:\n"
            + "\n".join(f"- {p}" for p in problems)
        )
        try:
            data = _extract_json(model.invoke(message).content)
        except (ValueError, json.JSONDecodeError) as exc:
            problems = [f"reply was not valid JSON ({exc})"]
            continue
        title = str(data.get("title") or "").strip()
        cleaned, problems = _validate(data.get("clues") or {}, answers)
        if not problems:
            break

    missing = [slot_id for slot_id in answers if slot_id not in cleaned]
    if missing:
        raise RuntimeError(f"model never clued: {', '.join(missing)}")
    return title or "Freshly generated puzzle", cleaned


def to_puzzle(name: str, grid: Grid, title: str, clues: dict[str, str]) -> Puzzle:
    """Package a filled grid and slot-keyed clues into a solvable Puzzle."""
    by_direction: dict[str, dict[str, str]] = {"across": {}, "down": {}}
    for slot_id, clue in clues.items():
        slot = grid.slots[slot_id]
        by_direction[slot.direction][str(slot.number)] = clue
    return Puzzle(
        id=f"generated-{name}",
        title=title,
        grid_rows=list(grid.template),
        clues=by_direction,
        solution=grid.render().split("\n"),
    )


def generate_puzzle(model, rng: random.Random | None = None, template: str | None = None) -> Puzzle:
    """Fill a grid, clue it with the model, and return a solvable Puzzle."""
    name, grid = fill_random_template(rng=rng, template=template)
    title, clues = write_clues(model, grid)
    return to_puzzle(name, grid, title, clues)
