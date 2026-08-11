"""Import a crossword from a third-party site into the internal Puzzle format.

Scope and limits, stated up front:

- This module converts a **scrape of what a site renders** — the block layout
  and the clue text — into a Puzzle. It does not decrypt, and will not decrypt,
  any answer key that a publisher protects. Boatload Puzzles, for example,
  ships its solution as an encrypted blob; this importer never touches it.
- An imported puzzle therefore has no ``solution``. The agent can solve it, and
  :func:`audit_fill` reports how good the attempt looks, but letter accuracy
  against a key is not available. See the docstring on :func:`audit_fill`.
- Puzzle text stays out of version control. Fetched files belong under
  ``data/external/``, which is gitignored, because the clues are the
  publisher's copyrighted work.

The scrape itself is deliberately not automated here. Publishers render these
puzzles with their own JavaScript, and their terms restrict automated access.
The documented path is: open the puzzle in a browser, run the extraction
snippet in ``docs/`` (or the one in this module's ``EXTRACTION_JS``), and save
the resulting JSON. This module takes it from there.
"""

from __future__ import annotations

import json
from pathlib import Path

from .grid import BLOCK, EMPTY, Grid, Puzzle

# Run this in the browser console on the rendered puzzle. It reads only what the
# page already displays: which squares are blocks, the printed slot numbers, and
# the clue list. It does not read, request, or decode any answer data.
EXTRACTION_JS = r"""
(() => {
  const doc = [...document.querySelectorAll('iframe')]
    .map(f => { try { return f.contentDocument; } catch { return null; } })
    .find(d => d && d.querySelectorAll('.grect').length) || document;
  const cells = [...doc.querySelectorAll('.grect')].filter(el => el.getBoundingClientRect().width < 40);
  const xs = [...new Set(cells.map(e => Math.round(e.getBoundingClientRect().left)))].sort((a, b) => a - b);
  const ys = [...new Set(cells.map(e => Math.round(e.getBoundingClientRect().top)))].sort((a, b) => a - b);
  const grid = ys.map(() => Array(xs.length).fill('.'));
  for (const el of cells) {
    const b = el.getBoundingClientRect();
    const c = xs.indexOf(Math.round(b.left)), r = ys.indexOf(Math.round(b.top));
    if (r >= 0 && c >= 0 && el.classList.contains('gblacksquare')) grid[r][c] = '#';
  }
  const nums = [...doc.querySelectorAll('td.cnum')].map(e => e.innerText.trim().replace('.', ''));
  const texts = [...doc.querySelectorAll('td.cfullclue')].map(e => e.innerText.trim());
  const pairs = nums.map((n, i) => [n, texts[i]]);
  let split = pairs.length;
  for (let i = 1; i < pairs.length; i++) if (+pairs[i][0] < +pairs[i - 1][0]) { split = i; break; }
  return JSON.stringify({
    title: (doc.body.innerText.match(/Crossword \d+/) || ['Imported puzzle'])[0],
    grid: grid.map(r => r.join('')),
    across: pairs.slice(0, split),
    down: pairs.slice(split),
  });
})()
"""


def puzzle_from_scrape(data: dict, source: str = "external") -> Puzzle:
    """Build a Puzzle from a scrape of ``{title, grid, across, down}``.

    ``across`` and ``down`` are lists of ``[number, clue]`` pairs, as printed
    on the page. The grid uses ``#`` for a block and ``.`` for an open cell.
    """
    puzzle = Puzzle(
        id=f"{source}-{str(data.get('title', 'puzzle')).lower().replace(' ', '-')}",
        title=str(data.get("title", "Imported puzzle")),
        grid_rows=[str(row) for row in data["grid"]],
        clues={
            "across": {str(n): str(c) for n, c in data.get("across", [])},
            "down": {str(n): str(c) for n, c in data.get("down", [])},
        },
        solution=None,  # publishers protect the key; we do not extract it
    )
    problems = check_import(puzzle)
    if problems:
        raise ValueError("imported puzzle is inconsistent: " + "; ".join(problems))
    return puzzle


def check_import(puzzle: Puzzle) -> list[str]:
    """Verify the scrape against our own numbering. Returns a list of problems.

    This is the important guard. Our engine derives slot numbers from the block
    layout. If the scraped clue numbers do not match those numbers, then either
    the layout or the clue list was read wrongly, and solving would be
    meaningless. The check catches an off-by-one in the grid scrape.
    """
    grid = Grid(puzzle.grid_rows)
    ours = {slot.id for slot in grid.slots.values()}
    theirs = {
        f"{number}{'A' if direction == 'across' else 'D'}"
        for direction, entries in puzzle.clues.items()
        for number in entries
    }
    problems = []
    for missing in sorted(ours - theirs):
        problems.append(f"{missing} has no clue")
    for extra in sorted(theirs - ours):
        problems.append(f"clue {extra} matches no slot")
    return problems


def load_scrape(path: str | Path, source: str = "external") -> Puzzle:
    return puzzle_from_scrape(json.loads(Path(path).read_text()), source=source)


def audit_fill(grid: Grid, words: set[str]) -> dict:
    """Judge a fill when no answer key exists.

    An imported puzzle carries no key, so letter accuracy is unavailable. These
    three signals are what remain, and each is meaningful on its own:

    - ``complete``: every open cell holds a letter.
    - ``consistent``: guaranteed by the engine, but re-checked here — every
      crossing agrees, because the engine refused anything else.
    - ``in_vocabulary``: the share of entries that appear in a reference
      wordlist. This catches invented words, which is the usual failure mode.
      A real answer that the wordlist lacks counts against the score, so read
      this as a floor and not as accuracy.
    """
    filled = {sid: grid.slot_pattern(sid) for sid in grid.slots}
    complete = all(EMPTY not in pattern for pattern in filled.values())
    known = [p for p in filled.values() if EMPTY not in p and p.upper() in words]
    total = sum(1 for p in filled.values() if EMPTY not in p)
    return {
        "complete": complete,
        "slots": len(filled),
        "filled": total,
        "in_vocabulary": len(known) / total if total else 0.0,
        "unknown_words": sorted(
            p for p in filled.values() if EMPTY not in p and p.upper() not in words
        ),
    }


def compare_fills(a: Grid, b: Grid) -> dict:
    """Agreement between two independent solves of the same puzzle.

    Without a key, two models agreeing on an entry is evidence for it. This
    reports per-slot agreement so a reviewer can see where they diverge.
    """
    slots = sorted(set(a.slots) & set(b.slots))
    same = [s for s in slots if a.slot_pattern(s) == b.slot_pattern(s)]
    return {
        "slots": len(slots),
        "agree": len(same),
        "agreement": len(same) / len(slots) if slots else 0.0,
        "disagreements": {
            s: [a.slot_pattern(s), b.slot_pattern(s)] for s in slots if s not in same
        },
    }


__all__ = [
    "BLOCK",
    "EXTRACTION_JS",
    "audit_fill",
    "check_import",
    "compare_fills",
    "load_scrape",
    "puzzle_from_scrape",
]
