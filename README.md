# Nebius-XWord

An LLM crossword-solving agent built on [Nebius AI Studio](https://studio.nebius.com).

The agent runs a tool-use loop: the model inspects the grid and clues, fills the
slots it is confident about, uses crossing letters to constrain the rest, and
backtracks when the grid reports conflicts. A deterministic grid engine enforces
the rules (slot detection, numbering, conflict checking), so the model never has
to be trusted about grid state — only about answers.

## Repository layout

```
├── src/nebius_xword/     # core package
│   ├── grid.py           #   grid model: slots, numbering, fills, conflicts
│   ├── tools.py          #   tool schemas + executor exposed to the LLM
│   ├── agent.py          #   CrosswordAgent: tool loop on Nebius AI Studio
│   └── solver.py         #   deterministic utilities (pattern match, validation)
├── data/puzzles/         # example puzzles (JSON, with answer keys)
├── eval/                 # evaluation harness + metrics
├── scripts/              # experiment entry points
├── tests/                # pytest suite (no API key needed)
└── .env.example          # Nebius AI Studio configuration template
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then paste your Nebius AI Studio API key
```

## Run

Solve one puzzle with the agent (prints the tool-call trace, final grid, and score):

```bash
python scripts/solve.py data/puzzles/example_mini_5x5.json
```

Run the eval harness over all puzzles in `data/puzzles/`:

```bash
python -m eval.run_eval --solver llm
```

The harness also has two no-API sanity solvers — `--solver oracle` (fills from
the answer key; must score 100%) and `--solver empty` (must score 0%) — useful
for validating the harness itself. Metrics reported per puzzle and in aggregate:
**letter accuracy**, **word accuracy**, and **solved** (fully correct grid).

## Tests

```bash
pytest
```

The suite covers grid numbering, conflict detection, puzzle-file validation, and
metric scoring. It needs no API key.

## Puzzle format

```json
{
  "id": "example_mini_5x5",
  "title": "Mini crossword (5x5)",
  "grid":     ["##...", "#....", ".....", "....#", "...##"],
  "clues":    {"across": {"1": "Peach center"}, "down": {"1": "..."}},
  "solution": ["##PIT", "#TIRE", "SALON", "UPON#", "NET##"]
}
```

`#` is a block, `.` an open cell. Slot numbers are derived from the grid using
standard crossword numbering, so clue keys must match that numbering.
`solution` is optional; puzzles without it can be solved but not scored.

## Roadmap

- [ ] Wordlist-backed backtracking solver in `solver.py` as a non-LLM baseline
      and as a repair step for near-complete LLM fills
- [ ] Candidate-suggestion tool (`suggest(pattern)`) so the model can query the
      wordlist for slots it can't answer from the clue alone
- [ ] Puzzle importers (.puz / NYT-style formats) to grow the eval set
- [ ] Per-model eval comparison across Nebius AI Studio hosted models
