# Nebius-XWord

An AI agent that solves crossword puzzles, with a deterministic grid engine, an
evaluation harness, and a web UI.

**Live demo: <https://nebius-xword.vercel.app>** — pick a puzzle, pick a model,
press Solve, and watch the agent fill the grid (the Oracle option demos the
pipeline without spending tokens).

The agent is a **LangGraph** `StateGraph` with two nodes: an *agent* node
(the LLM, bound to four tools) and a *tools* node backed by a deterministic
grid engine. The model inspects the grid and clues, fills the slots it is
confident about, uses crossing letters to constrain the rest, and backtracks
when the grid reports conflicts. The run ends when the model calls `submit`,
stops calling tools, or hits the turn cap. The grid engine enforces the rules
(slot detection, numbering, conflict checking), so the model is never trusted
about grid state — only about answers.

## Repository layout

```
├── src/nebius_xword/     # core package (dependency-free except agent.py)
│   ├── grid.py           #   grid model: slots, numbering, fills, conflicts
│   ├── tools.py          #   tool schemas + executor exposed to the LLM
│   ├── graph.py          #   LangGraph StateGraph: agent + tools nodes, stops
│   ├── agent.py          #   CrosswordAgent: model config + graph invocation
│   ├── generator.py      #   fresh puzzles: wordlist fill + LLM-written clues
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
| **Vercel AI Gateway** (default) | `LLM_API_KEY` (+ optional `LLM_MODEL`, `LLM_BASE_URL`) | `deepseek/deepseek-v4-flash-0731` |
| **Nebius AI Studio** (direct) | `NEBIUS_API_KEY` (+ optional `NEBIUS_MODEL`, `NEBIUS_BASE_URL`) | `meta-llama/Meta-Llama-3.1-70B-Instruct` |

The default is the dated DeepSeek checkpoint so eval numbers stay
reproducible; the undated alias `deepseek/deepseek-v4-flash` tracks weight
updates and is also allowlisted.

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

## Generating fresh puzzles

The demo can build a puzzle that has never existed and then solve it blind.
Work is split so each side does what it is reliable at:

1. **The grid is filled by search, not by the model.** `generator.py` picks a
   block template and runs the backtracking filler in `solver.py` over
   `data/wordlist.txt` (~1.7k common words). Every entry is therefore a real
   word and every crossing is consistent by construction — the model cannot
   invent a broken puzzle. Typical fill: under 0.1s for a 5×5, ~1s for the 7×7.
2. **The model writes the clues**, plus a title, in one JSON reply. Clues are
   validated: any clue that contains its own answer is masked to `___` (which
   is a legitimate fill-in-the-blank clue), and a retry is issued.
3. **The solve is blind.** The agent gets a fresh, empty grid and the clues.
   The answer key never enters its context; it is used afterwards, only to
   score the attempt.

Generating and solving are two separate requests (`POST /api/generate`, then
`POST /api/solve/stream` with the puzzle inline). Together they exceed a
single function timeout, so the browser holds the finished puzzle between the
two calls and hands it back. The answer key rides along in that hand-off
purely so the result can be scored; it is never part of what the solver reads.

**Latency is the honest weak point.** DeepSeek v4 Flash reasons at length, and
measured wall-clock varies widely: clue writing has taken 54s to 206s, and a
blind solve of a generated 5×5 about 150s. Two lessons came out of measuring
it. Telling the agent to open with `get_state` wasted a whole turn, because
the opening message already contains the grid — and a separate "confirm before
submitting" turn cost over 200s to re-verify a grid the engine had already
validated. Removing both cut a solve from roughly 420s to 153s. Generation
still occasionally approaches the 300s function ceiling; if it does, the page
says so and you press the button again.

Caveat worth stating plainly: the same model writes the clues and solves them.
It cannot see the answers while solving, but its own phrasing may suit it
better than a stranger's would. Treat generated-puzzle scores as a
demonstration, and the fixed puzzle set below as the real measurement.

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

`deepseek/deepseek-v4-flash-0731` (the default; LangGraph loop):

| puzzle | letters | words | solved | turns | tokens |
|---|---|---|---|---|---|
| example_3x3 | 100% | 100% | 2/2 | 3.5 | 5.9k |
| example_mini_5x5 | 100% | 100% | 2/2 | 3.5 | 15.1k |
| example_5x5_b | 100% | 100% | 2/2 | 3.5 | 9.8k |
| example_7x7 | 100% | 100% | 2/2 | 5.0 | 30.7k |

After the prompt was trimmed for latency (see below), a single re-run of the
same four puzzles still solved every one, in fewer turns: 2, 3, 2 and 2 turns,
taking 18s, 57s, 123s and 85s. The 7×7 dropped from 5 turns to 2.

At ~$0.014/M input and $0.028/M output, a full 4-puzzle eval run costs well
under a cent. `anthropic/claude-sonnet-4.5` also scored 8/8 on the
pre-LangGraph loop (3.5–5.5 turns, 6.4k–22.5k tokens) and was re-verified
post-port on the 7×7 (solved, 6 turns) — the port did not regress scores.

For contrast, `openai/gpt-4o-mini` on the 3×3 (1 run, pre-port) scored 67%
letters / 33% words — it answered SIX for "Half a score" and submitted despite
nonsense crossings, which is exactly the failure mode the word-accuracy metric
and the backtrack baseline are designed to expose.

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
  models from the `ALLOWED_MODELS` list in `api/index.py` — currently the two
  DeepSeek v4 Flash ids. Any gateway model still works locally through
  `LLM_MODEL` or `--model`; the allowlist only guards the hosted demo.

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
- [x] LangGraph port of the agent loop
- [ ] Candidate-suggestion tool (`suggest(pattern)`) so the model can query the
      wordlist for slots it can't answer from the clue alone
- [ ] Top-k candidate lists per clue + deterministic beam-search assignment
      (keeps the LLM out of the inner loop for big grids)
- [ ] JSON tool-protocol fallback for models without native tool calling
- [ ] Puzzle importers (.puz / NYT-style formats) to grow the eval set
- [ ] Per-model eval comparison across gateway-hosted models
