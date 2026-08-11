# xWord Agent

I built this project for the Nebius Forward Deployed Engineer assignment. The
assignment asks for three things: an AI agent that solves crossword puzzles, a
method to evaluate its solutions, and a repository with clear instructions.
This README explains what I did, why I did it, and how you can check each
claim yourself.

Start with the live demo: **<https://nebius-xword.vercel.app>**

Press Solve, and the agent fills the grid while a log shows each move. Press
Race, and both services run the same model at the same instant. Press
Generate, and the demo builds a puzzle that did not exist before — then both
services solve it blind.

## The design in one paragraph

I split the problem in two. A Python engine owns the grid: it finds the slots,
numbers them, and refuses any answer that breaks a crossing word. The model
owns the answers, and nothing else. This split is the core decision. A wrong
answer stays a wrong answer, and it can never become a broken grid. Everything
else in the repository exists to measure the agent honestly. Baselines bracket
its score. The generator cannot build an invalid puzzle. The race compares
providers with the model held constant.

## How the agent works

The agent is a [LangGraph](https://langchain-ai.github.io/langgraph/) graph
with two nodes. One node calls the model. The other node applies the model's
moves to the grid. I chose a hand-built graph over the prebuilt agent for one
reason. My stop condition is different: the run must end on a `submit` tool
call, not when the model goes quiet.

The model has four tools:

| Tool | Effect |
|---|---|
| `get_state` | Shows the grid, the clues, and the letters found so far. |
| `fill_slot` | Writes an answer into a slot. |
| `clear_slot` | Empties a slot. |
| `submit` | Ends the run. |

When an answer disagrees with a crossing word, the engine rejects it and gives
the reason. The model reads the reason and tries again. The run stops when the
model submits, stops calling tools, or reaches its turn limit.

I also measured the prompt, because speed matters in a demo. Telling the agent
to call `get_state` first wasted a full turn — the first message already
contains the grid. A "confirm before submit" instruction cost 200 seconds to
re-check a grid the engine had already validated. I removed both. One solve
fell from about 420 seconds to 153.

## Repository layout

```
├── src/nebius_xword/     # core package (only agent.py needs the LLM libraries)
│   ├── grid.py           #   the grid: slots, numbering, fills, conflicts
│   ├── tools.py          #   the four tools, and the code that runs them
│   ├── graph.py          #   the LangGraph graph: two nodes and the stop rules
│   ├── agent.py          #   CrosswordAgent: model setup, and the solve loop
│   ├── generator.py      #   new puzzles: grid search, then clues from the model
│   └── solver.py         #   pattern matching, grid filler, puzzle validation
├── api/index.py          # FastAPI app, and the entry point on Vercel
├── public/index.html     # the web page
├── data/puzzles/         # four example puzzles, each with its answer key
├── data/wordlist.txt     # 1,663 common words, used to build new grids
├── eval/                 # the evaluation harness and its metrics
├── scripts/              # command line entry points
├── tests/                # the test suite; no API key is necessary
├── vercel.json           # deployment settings
└── requirements.txt      # the packages Vercel installs
```

## Two services, one agent

I run the agent on Nebius Token Factory by default, and on Vercel AI Gateway
for comparison. To be precise about how:

- Both services expose the OpenAI chat API. The agent code that talks to a
  model is identical for both.
- What differs is configuration, and I wrote that routing deliberately. Each
  model **id** belongs to exactly one service. `connection_for()` in
  `api/index.py` maps an id to its service's host and key, so one request can
  never inherit the other service's credentials.
- The same weights appear on both services under different ids. That is the
  point: it makes every cross-service comparison a comparison of
  infrastructure, not of models.

| Service | Model id on the page | Price in/out per 1M tokens | Key |
|---|---|---|---|
| **Nebius Token Factory** (default) | `deepseek-ai/DeepSeek-V4-Pro` | $1.75 / $3.50 | `NEBIUS_API_KEY` |
| Vercel AI Gateway | `deepseek/deepseek-v4-pro` | $1.74 / $3.48 | `LLM_API_KEY`, or the Vercel OIDC token |

The dropdown on the page lists these two ids. Choosing an id therefore also
chooses the service that serves it. The model costs almost the same on both
services, which makes the race below a clean comparison.

Nebius AI Studio is now called **Nebius Token Factory**. Its host is
`https://api.tokenfactory.nebius.com/v1`, and keys come from
<https://tokenfactory.nebius.com>. On Vercel the gateway needs no key, because
the deployment sends its own OIDC token.

## Why these Nebius models

I did not guess the models — I read the live catalog and measured candidates:

```bash
curl -s -H "Authorization: Bearer $NEBIUS_API_KEY" \
  "https://api.tokenfactory.nebius.com/v1/models?verbose=true"
```

The agent works entirely through tool calls, so a model must support them. Of
28 models in the catalog, 23 report `tools`. One trap is worth recording:
`deepseek-ai/DeepSeek-V4-Flash` does **not** support tools on Nebius, so the
cheap obvious choice cannot drive this agent. I found that in the catalog
before it could fail in production.

I then ran candidates on the sample puzzles:

| Model | Result |
|---|---|
| `deepseek-ai/DeepSeek-V4-Pro` | Solved every puzzle. My default. |
| `nvidia/nemotron-3-super-120b-a12b` | Solved the two smaller puzzles only. |
| `meta-llama/Llama-3.3-70B-Instruct` | Solved the 3×3, but needed 7 turns. |
| `openai/gpt-oss-120b` | Failed the 3×3 at 33% of letters. |

Only DeepSeek V4 Pro solves the whole set, so it is the only model the page
offers. Several faster or cheaper models solve the 3×3 and then fail a 7×7,
which is worse than useless in a demo: the speed looks good until the grid
does not finish. I measured the candidates rather than guessing, and I record
the failures as well as the winner, because the failures are what a colleague
needs to know.

## Run it yourself

Requires Python 3.10 or newer. Run every command from the repository root.

### 1. Install

```bash
git clone https://github.com/jaredwerba/Nebius-XWord.git
cd Nebius-XWord
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Check it works — no API key needed

These three commands need no key and no network. Run them first to prove the
install is good.

```bash
pytest                                    # 65 tests
python -m eval.run_eval --solver oracle   # must print 100% everywhere
python -m eval.run_eval --solver empty    # must print 0% everywhere
```

### 3. Add a key

```bash
cp .env.example .env
```

Open `.env` and set **one** of these:

- `NEBIUS_API_KEY` — get one at <https://tokenfactory.nebius.com>. This is the
  default path, and the one all the results below use.
- `LLM_API_KEY` — a Vercel AI Gateway key, if you prefer that provider.

### 4. Solve a puzzle

```bash
python scripts/solve.py data/puzzles/example_mini_5x5.json
```

It prints each tool call as the agent makes it, then the finished grid, the
model, the turns, the tokens, and the score:

```
model: deepseek-ai/DeepSeek-V4-Pro | turns: 2 | tokens: 3541 | submitted: True
score: {'letter_accuracy': 1.0, 'word_accuracy': 1.0, 'solved': True}
```

Add `--model <id>` to pick a model, or `--quiet` to hide the trace.

### 5. Score the agent on the whole puzzle set

```bash
python -m eval.run_eval --solver llm --runs 3
```

See [Evaluation methodology](#evaluation-methodology) for what the numbers
mean.

### 6. Run the web app

```bash
uvicorn api.index:app --reload
```

Open <http://127.0.0.1:8000>. Press Solve to watch the agent work, Race to
run two providers at once, or Generate to build a new puzzle and race it
blind.

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: nebius_xword` | The install step did not run, or the virtual environment is not active. Re-run `pip install -e ".[dev]"`. |
| `401` from the model provider | The key in `.env` is missing, wrong, or expired. Only one of `NEBIUS_API_KEY` or `LLM_API_KEY` is needed. |
| `model must be one of [...]` | The hosted demo restricts models by design. Locally, use `--model` with any id your provider serves. |
| A solve stops early with an unfinished grid | The turn cap was reached. Raise it in `CrosswordAgent(max_turns=...)`; big puzzles need more turns. |

## New puzzles, solved blind

The Generate button builds a puzzle that did not exist before. I split the
work so each side does what it is reliable at:

1. **A search fills the grid, not the model.** `generator.py` picks a block
   layout, and the backtracking filler in `solver.py` fits words from
   `data/wordlist.txt`. Every entry is a real word, and every crossing agrees.
   The model cannot produce a broken puzzle, because the model does not build
   the grid. A 5×5 fills in under 0.1 seconds; the 7×7 takes about 1 second.
2. **The model writes the clues** — DeepSeek V4 Pro on Nebius — in one JSON
   reply, with a title. I validate each clue: if a clue contains its own
   answer, the code masks the answer to `___` and asks again. On Nebius, clue
   writing has taken 3 to 52 seconds. On the gateway model I used earlier, it
   took 52 to 206 seconds.
3. **Both services then race to solve it blind.** Each agent starts from an
   empty grid and receives the clues only. The answer key never enters a
   prompt. It is used once, at the end, to score both attempts.

Generation and solving are separate requests, because together they can exceed
one function timeout. The browser holds the finished puzzle between the calls.
The answer key travels in that hand-off only so the result can be scored.

One caveat, stated plainly: the same model family writes the clues and then
solves them. It cannot see the answers, but its own phrasing may suit it. Read
generated-puzzle scores as a demonstration. The fixed puzzle set below is the
measurement.

## The race

The Race button answers one question: with the model held constant, which
service returns the answer sooner? The browser fires two identical requests in
the same tick. Both run DeepSeek V4 Pro — the same weights under each
service's id — so the clocks compare infrastructure. Two cards stream the
moves with their own timers. The verdict names the winner, both times, the
ratio, and whether both grids agree.

Each race adds a row to a Speed comparison chart: one bar per service, on a
shared scale, with the times on the bars. I chose a two-bar chart because the
data is two durations. I validated the bar colors for contrast and
color-vision separation, with a script and not by eye.

Races I recorded while building: 8.9s vs 25.0s, 11.8s vs 24.8s, 19.5s vs
77.8s, and a generated puzzle at 29.8s vs 87.3s. Nebius won each one, by 2.1×
to 4.0×. The page shows the recorded average as a permanent chart: Nebius
17.5s, the gateway 53.7s — Nebius 3.1× faster on the same DeepSeek V4 Pro
weights. New races add rows below that average. Treat any single race as one sample: the clock starts in the
browser, includes the network and any cold start, and providers share load.
That honesty is deliberate — the chart shows measurements, not marketing.

## A real external puzzle

The sample puzzles are small, so I pointed the agent at a full
newspaper-size crossword: the daily 13×13 from boatloadpuzzles.com, with 60
interlocking entries. `src/nebius_xword/external.py` imports it from what the
page displays — the block layout and the clue text. The import validates
itself: the engine derives the slot numbers from the blocks, and all 60
matched the numbers printed on the page.

**DeepSeek V4 Pro on Nebius completed the full grid and submitted it**: 60 of
60 entries, every crossing verified by the engine, in 98 turns, 16.6 minutes,
and 2.44 million tokens — about $4.40 of inference. Two agent improvements
came out of this run. A history window bounds what the model re-reads each
turn, which keeps token cost linear instead of quadratic. And a nudge step
pushes a model back to tool calls when it answers in prose, which ended one
early attempt at turn one.

Other models fell short, and I report that plainly. Smaller models ran out of
turns with the grid unfinished. GLM-5.1 works at a pace that would need about
an hour for this grid, so I stopped it — a demo should not make a reviewer
wait.

One boundary matters here. The publisher encrypts its answer key, and this
project does not break encryption. "Completed" therefore means the grid is
full and every crossing agrees — a strong structural constraint on 60
interlocking entries, but not a score against the key. Imported puzzle text
stays out of the repository, because it is the publisher's copyrighted work.

## Evaluation methodology

This is the deliverable I care most about, because "the agent solved it" is
easy to claim and hard to trust. The harness is `eval/run_eval.py`, and the
scoring is `eval/metrics.py`.

### What I measure, and why

Each metric answers a different question. I report all four, because any one
of them alone can flatter the agent.

| Metric | Definition | Why it is here |
|---|---|---|
| **Letter accuracy** | correct open cells ÷ total open cells | Partial credit. Shows progress when the grid is close but not finished. |
| **Word accuracy** | fully correct slots ÷ total slots | Stricter. One wrong letter voids the whole entry, which is how a crossword actually works. |
| **Solved rate** | runs with a fully correct grid ÷ runs | The headline. A crossword is a pass-or-fail artifact. |
| **Cost** | mean turns and tokens per solve | Quality at any price is not a result. This is what makes it an engineering number. |

Letter accuracy and word accuracy can disagree sharply, and the gap is
informative. A model that guesses plausible words with one wrong letter each
scores well on letters and near zero on words. That is exactly the failure
`openai/gpt-4o-mini` produced on the 3×3 — 67% of letters, 33% of words.

### How to reproduce it

No API key is needed for the first two commands.

```bash
python -m eval.run_eval --solver oracle      # must print 100% everywhere
python -m eval.run_eval --solver empty       # must print 0% everywhere
python -m eval.run_eval --solver backtrack   # about 9% of letters
python -m eval.run_eval --solver llm --runs 3
```

Output looks like this:

```
solver: oracle | runs per puzzle: 1

puzzle                letters    words   solved
example_3x3             100%     100%     1/1
example_5x5_b           100%     100%     1/1
example_7x7             100%     100%     1/1
example_mini_5x5        100%     100%     1/1

mean / total            100%     100%     4/4
```

Add `--json` for machine-readable output, `--model <id>` to test one model,
and `--puzzles <dir>` to point at your own puzzles.

### Baselines: bracketing the score, and testing the test

Four solvers run through the same scoring code. Two of them exist to prove
the harness is not lying.

| Solver | What it does | Expected | Purpose |
|---|---|---|---|
| `empty` | Leaves the grid blank. | 0% | Floor. If this is not 0, the scorer is broken. |
| `backtrack` | Fits real dictionary words, ignores the clues. | ~9% letters | Structure-only baseline. |
| `llm` | The agent under test. | — | The measurement. |
| `oracle` | Copies the answer key. | 100% | Ceiling. If this is not 100, the scorer is broken. |

`backtrack` is the baseline I think matters most. It fills the grid with real
words that interlock correctly, and it never reads a clue. It scores about 9%
of letters. Any score above that line is the part the language model
contributed by understanding clues. Without this baseline, a grid full of
valid-looking words could be mistaken for comprehension.

### How to read the numbers

- **Repeat the runs.** Model output is not deterministic. `--runs 3` is the
  minimum I trust for a comparison; single runs are for smoke tests.
- **Four puzzles is a small sample.** These results indicate capability. They
  do not establish a statistically tight ranking between two strong models.
- **Both services solve every fixed puzzle**, so on this set accuracy does not
  separate them. Speed does. That is a limit of the puzzle set, not a finding
  about the models.

### Evaluating without an answer key

A real newspaper puzzle does not ship its solution, so `score_grid` cannot
run. `src/nebius_xword/external.py` provides two substitutes:

- **`audit_fill`** reports completeness (are all slots filled), crossing
  consistency (the engine verifies every shared letter), and the share of
  entries that appear in a reference dictionary, with the unknown words
  listed.
- **`compare_fills`** reports per-slot agreement between two independent
  solves. Agreement between different model families is weak evidence of
  correctness; disagreement localises exactly which entries are contested.

This is a deliberately weaker claim than a keyed score, and the README says so
wherever those numbers appear.

### What this does not measure

Being explicit about the edges: the harness does not judge clue quality on
generated puzzles, does not handle themed or rebus puzzles, is English-only,
and assumes each answer key is the single correct solution. Solve latency is
measured separately by the race, because it depends on the provider rather
than on the agent.

### The puzzle set

Four hand-checked puzzles: a 3×3 smoke test, two 5×5s (one plain, one with
wordplay), and a 7×7 pinwheel. Every puzzle carries an answer key, and the
test suite validates the numbering, the clue coverage, and the key itself, so
a malformed puzzle fails CI rather than quietly scoring badly.

## Results

**Nebius Token Factory, `deepseek-ai/DeepSeek-V4-Pro`** — two runs per puzzle,
10 August 2026. All 8 runs solved.

| Puzzle | Letters | Words | Solved | Turns | Tokens | Time |
|---|---|---|---|---|---|---|
| example_3x3 | 100% | 100% | 2/2 | 2.0 | 3.2k | 5.9s |
| example_mini_5x5 | 100% | 100% | 2/2 | 10.5 | 48.8k | 39.1s |
| example_5x5_b | 100% | 100% | 2/2 | 5.0 | 14.7k | 19.0s |
| example_7x7 | 100% | 100% | 2/2 | 5.5 | 24.9k | 39.5s |

**Vercel AI Gateway, `deepseek/deepseek-v4-flash-0731`** — the same test, run
when V4 Flash was the page's gateway model. Also 8/8. The page now mirrors the
Nebius models instead, so this table stays as history.

| Puzzle | Letters | Words | Solved | Turns | Tokens |
|---|---|---|---|---|---|
| example_3x3 | 100% | 100% | 2/2 | 3.5 | 5.9k |
| example_mini_5x5 | 100% | 100% | 2/2 | 3.5 | 15.1k |
| example_5x5_b | 100% | 100% | 2/2 | 3.5 | 9.8k |
| example_7x7 | 100% | 100% | 2/2 | 5.0 | 30.7k |

Both services solve every fixed puzzle, so accuracy does not separate them.
The set is too easy to rank two strong models — a limit of the set, not a
finding. Speed does separate them; see the race section.

Costs stay small. The Nebius runs above used about 180k tokens, roughly 40
cents at V4 Pro prices. The historical Flash runs cost fractions of a cent at
$0.014 / $0.028 per million.

Two other models ran on the earlier loop. `anthropic/claude-sonnet-4.5`
solved all 8 of its runs. `openai/gpt-4o-mini` failed its one run — it
answered SIX for "Half a score" and submitted a grid with broken crossings.
Word accuracy and the `backtrack` baseline exist to expose exactly that
behavior.

## Tests

```bash
pytest
```

65 tests, and none needs an API key or the network. A scripted fake model
drives the whole LangGraph loop offline, so the stop rules, the token counts,
the history window, and the tool dispatch are tested on every run. Other
tests pin the page's model ids to the server's, so the dropdown and the
allowlist cannot drift apart silently. The importer is covered with synthetic
fixtures, so no third-party puzzle is needed to test it.

## Deployment

The app runs on Vercel at <https://nebius-xword.vercel.app>. A push to `main`
deploys; `vercel deploy --prod` also deploys.

Details a reviewer may care about:

- Vercel serves `public/` directly. `vercel.json` sends the five `/api/*`
  routes to the FastAPI function, with the original path carried in a `__path`
  parameter, because the Python runtime sees only the rewrite destination.
  Middleware restores the path. `maxDuration` is 300 seconds.
- The gateway authenticates with the deployment's own OIDC token, taken from
  the request by middleware. Nebius uses `NEBIUS_API_KEY`, set as a project
  variable.
- The race found a real bug here: the first request in a fresh process read
  the environment before `.env` loaded, and the fallback chain handed the
  gateway request the Nebius key. I load the environment at startup now, and
  a test covers the routing. I record this because finding it required firing
  both services at once — single-service tests passed.
- The public endpoints spend real credit, so `ALLOWED_MODELS` in
  `api/index.py` restricts them to the four ids on the page. Locally,
  `LLM_MODEL`, `NEBIUS_MODEL`, and `--model` accept anything either service
  offers.

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

`#` is a block. `.` is an open cell. Slot numbers follow the standard
crossword rules, and `solver.verify_puzzle` checks that the clue keys agree
with them. `solver.fill_grid` builds new grids — the same filler the
generator uses.

## Next steps

- [x] Wordlist filler, for the baseline and for new grids
- [x] LangGraph agent loop
- [x] New puzzles, generated and then raced blind
- [x] Two services with identical weights, and a measured race
- [ ] A `suggest(pattern)` tool, so the model can ask the wordlist for
      candidates when a clue defeats it
- [ ] Candidate lists per clue, and a beam search to choose between them
- [ ] Importers for `.puz` files, to grow the evaluation set
- [ ] A standing eval that reruns on a schedule and tracks drift
