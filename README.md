# Nebius-XWord

An AI agent that solves crossword puzzles, with a deterministic grid engine, an
evaluation harness, and a web UI.

**Live demo: <https://nebius-xword.vercel.app>** — pick a puzzle, pick a model,
press Solve, and watch the agent fill the grid (the Oracle option demos the
pipeline without spending tokens).

The agent runs a tool-use loop against an LLM: the model inspects the grid and
clues, fills the slots it is confident about, uses crossing letters to
constrain the rest, and backtracks when the grid reports conflicts. The grid
engine enforces the rules (slot detection, numbering, conflict checking), so
the model is never trusted about grid state — only about answers.

## Repository layout

```
├── src/nebius_xword/     # core package (dependency-free except agent.py)
│   ├── grid.py           #   grid model: slots, numbering, fills, conflicts
│   ├── tools.py          #   tool schemas + executor exposed to the LLM
│   ├── agent.py          #   CrosswordAgent: the LLM tool loop
│   └── solver.py         #   pattern matching, backtracking filler, validation
├── api/index.py          # FastAPI app (also the Vercel entry point)
├── public/index.html     # web UI: pick a puzzle, watch the agent solve it
├── data/puzzles/         # hand-verified example puzzles (JSON, with keys)
├── eval/                 # evaluation harness + metrics
├── scripts/              # CLI entry points
├── tests/                # pytest suite (no API key needed)
├── vercel.json           # deployment config (all routes -> the ASGI app)
└── requirements.txt      # runtime deps for Vercel (pyproject has the rest)
```

## LLM configuration

Two interchangeable OpenAI-compatible backends (see `.env.example`):

| | env vars | default model |
|---|---|---|
| **Vercel AI Gateway** (default) | `LLM_API_KEY` (+ optional `LLM_MODEL`, `LLM_BASE_URL`) | `openai/gpt-4o-mini` |
| **Nebius AI Studio** (direct) | `NEBIUS_API_KEY` (+ optional `NEBIUS_MODEL`, `NEBIUS_BASE_URL`) | `meta-llama/Meta-Llama-3.1-70B-Instruct` |

When deployed on Vercel, the function's OIDC token is used for AI Gateway
automatically, so no key needs to be configured. Explicit constructor args >
`LLM_*` > `NEBIUS_*`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then add your AI Gateway or Nebius key
```

## Run

**CLI** — solve one puzzle (prints the tool-call trace, final grid, and score):

```bash
python scripts/solve.py data/puzzles/example_mini_5x5.json
```

**Web UI** — same agent behind a FastAPI app:

```bash
uvicorn api.index:app --reload
```

then open http://127.0.0.1:8000. The `Oracle` solver in the UI demos the full
pipeline without an API key.

**Evaluation** — run a solver over every puzzle in `data/puzzles/`:

```bash
python -m eval.run_eval --solver llm --runs 3
```

## Evaluation methodology

**Metrics** (per puzzle, averaged over `--runs` repetitions because LLM solves
are stochastic):

- **Letter accuracy** — correct open cells / total open cells. Partial credit;
  the primary quality signal.
- **Word accuracy** — slots whose full answer is correct / total slots.
  Punishes near-misses that break crossings.
- **Solved rate** — fraction of runs with a fully correct grid. The headline
  number.
- **Cost** — mean agent turns and total tokens per solve (llm solver only).

**Baselines** bracket the agent's score and validate the harness itself:

| solver | what it does | expected score |
|---|---|---|
| `empty` | leaves the grid blank | 0% (floor) |
| `backtrack` | fills valid words from a wordlist, ignoring clues | ~10% letters (structure-only) |
| `llm` | the agent under evaluation | — |
| `oracle` | copies the answer key | 100% (ceiling) |

The gap between `backtrack` and `llm` isolates how much clue understanding
(not just grid-consistency search) the agent contributes.

**Puzzle set** — four hand-verified puzzles spanning difficulty: a 3×3 smoke
test, two 5×5 minis (one with straightforward clues, one with trickier
wordplay), and a 7×7 pinwheel. Every puzzle ships with an answer key and is
validated by CI-style tests (block symmetry, numbering, clue coverage, key
consistency).

**Known limitations** — small N (results are indicative, not statistically
tight); English-only; no themed/rebus puzzles; the eval trusts the answer key
as the unique solution.

### Results (2026-08-10, via the deployed agent, 2 runs per puzzle)

`anthropic/claude-sonnet-4.5`:

| puzzle | letters | words | solved | turns | tokens |
|---|---|---|---|---|---|
| example_3x3 | 100% | 100% | 2/2 | 3.5 | 6.4k |
| example_mini_5x5 | 100% | 100% | 2/2 | 4.0 | 9.4k |
| example_5x5_b | 100% | 100% | 2/2 | 3.5 | 8.3k |
| example_7x7 | 100% | 100% | 2/2 | 5.5 | 22.5k |

For contrast, `openai/gpt-4o-mini` on the 3×3 (1 run) scored 67% letters / 33%
words — it answered SIX for "Half a score" and submitted despite nonsense
crossings, which is exactly the failure mode the word-accuracy metric and the
backtrack baseline are designed to expose.

## Tests

```bash
pytest
```

Covers grid numbering, conflict detection, puzzle validation, metric scoring,
and the API endpoints (oracle path). No API key needed.

## Deployment (Vercel)

Deployed at <https://nebius-xword.vercel.app> (`vercel deploy --prod`; pushes
to `main` also auto-deploy). Notes:

- `public/` is served statically; `vercel.json` rewrites the three `/api/*`
  routes to the FastAPI function with the original path carried in a `__path`
  query param (Vercel's Python runtime only sees the rewrite destination path;
  middleware in `api/index.py` restores it). `maxDuration: 300` gives the
  agent loop room.
- **Auth is zero-config on Vercel**: the middleware picks up the request's
  Vercel OIDC token and uses it for AI Gateway. Set `LLM_API_KEY` /
  `LLM_MODEL` project env vars to override.
- Because the public endpoint spends gateway tokens, `/api/solve` only accepts
  models from the `ALLOWED_MODELS` list in `api/index.py`.

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

`#` is a block, `.` an open cell. Slot numbers derive from standard crossword
numbering, so clue keys must match it (`solver.verify_puzzle` checks this).
New puzzles can be constructed with `solver.fill_grid`, the wordlist-backed
backtracking filler.

## Roadmap

- [x] Wordlist-backed backtracking filler (baseline + puzzle construction)
- [ ] Candidate-suggestion tool (`suggest(pattern)`) so the model can query the
      wordlist for slots it can't answer from the clue alone
- [ ] Puzzle importers (.puz / NYT-style formats) to grow the eval set
- [ ] Per-model eval comparison across gateway-hosted models
