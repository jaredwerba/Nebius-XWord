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

| Service | Model ids on the page | Price in/out per 1M tokens | Key |
|---|---|---|---|
| **Nebius Token Factory** (default) | `deepseek-ai/DeepSeek-V4-Pro` · `Qwen/Qwen3-235B-A22B-Instruct-2507` | $1.75 / $3.50 · $0.20 / $0.60 | `NEBIUS_API_KEY` |
| Vercel AI Gateway | `deepseek/deepseek-v4-pro` · `alibaba/qwen-3-235b` | $1.74 / $3.48 · $0.22 / $0.88 | `LLM_API_KEY`, or the Vercel OIDC token |

The dropdown on the page lists these four ids. Choosing an id therefore also
chooses the service that serves it. V4 Pro costs almost the same on both
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
| `Qwen/Qwen3-235B-A22B-Instruct-2507` | Solved the 3×3 in 2.5s, but failed the harder puzzles. |
| `nvidia/nemotron-3-super-120b-a12b` | Solved the two smaller puzzles only. |
| `meta-llama/Llama-3.3-70B-Instruct` | Solved the 3×3, but needed 7 turns. |
| `openai/gpt-oss-120b` | Failed the 3×3 at 33% of letters. |

I kept two models on purpose. DeepSeek V4 Pro gives dependable tool calling.
Qwen3 235B gives raw speed. The two goals pull in different directions on this
catalog, and the pair covers both. I documented the failures as well as the
successes, because the failures are what a colleague needs to know.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Then put your Nebius key, or an AI Gateway key, in `.env`.

## Run

Solve one puzzle from the command line. The output shows each tool call, the
final grid, and the score.

```bash
python scripts/solve.py data/puzzles/example_mini_5x5.json
```

Start the web page, then open <http://127.0.0.1:8000>.

```bash
uvicorn api.index:app --reload
```

Score the agent on every puzzle in `data/puzzles/`:

```bash
python -m eval.run_eval --solver llm --runs 3
```

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
pushes a model back to tool calls when it answers in prose, which is how the
first Qwen attempt died.

The other models fell short, and I report that plainly. Qwen3 235B filled 36
of 60 entries before its 160-turn budget ran out. GLM-5.1 works at a pace
that would need about an hour for this grid, so I stopped it — a demo should
not make a reviewer wait.

One boundary matters here. The publisher encrypts its answer key, and this
project does not break encryption. "Completed" therefore means the grid is
full and every crossing agrees — a strong structural constraint on 60
interlocking entries, but not a score against the key. Imported puzzle text
stays out of the repository, because it is the publisher's copyrighted work.

## Evaluation

Model output varies between runs, so each metric is an average over the runs
you request:

- **Letter accuracy.** Correct open cells over total open cells. Partial
  credit.
- **Word accuracy.** Correct slots over total slots. An almost-right answer
  breaks its crossings, and this metric punishes that.
- **Solved rate.** Runs that produce a fully correct grid.
- **Cost.** Mean turns and tokens per solve.

Four solvers bracket the score, and two of them also prove the harness:

| Solver | What it does | Expected score |
|---|---|---|
| `empty` | Leaves the grid blank. | 0% — the floor |
| `backtrack` | Fits real words, ignores the clues. | about 10% of letters |
| `llm` | The agent under test. | — |
| `oracle` | Copies the answer key. | 100% — the ceiling |

The gap between `backtrack` and `llm` isolates clue understanding. That gap is
the thing the assignment actually asks about, so I built a baseline to expose
it.

The puzzle set has four hand-checked puzzles: a 3×3, two 5×5s (one plain, one
with wordplay), and a 7×7 pinwheel. Tests validate the numbering, the clue
coverage, and each answer key. The set is small, English-only, and has no
themed puzzles — read the numbers as an indication, not proof.

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

58 tests, and none needs an API key or the network. A scripted fake model
drives the whole LangGraph loop offline, so the stop rules, the token counts,
and the tool dispatch are tested on every run. Other tests pin the page's
model ids to the server's, so the dropdown and the allowlist cannot drift
apart silently.

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
