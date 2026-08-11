# Nebius-XWord

Nebius-XWord is an AI agent that solves crossword puzzles. It runs on **Nebius
Token Factory** inference by default, and it can run the same agent on Vercel
AI Gateway so you can compare the two.

Try it here: **<https://nebius-xword.vercel.app>**

Choose a puzzle and press Solve. The grid fills in while you watch, and a log
shows each move the agent makes. Or press Generate. The agent then builds a
new puzzle and solves it without the answers. The second dropdown chooses the
model, and with it the service that runs the model.

## How it works

The agent is a [LangGraph](https://langchain-ai.github.io/langgraph/) graph
with two nodes. The first node calls the model. The second node applies the
model's moves to the grid.

A Python engine owns the grid. The engine finds the slots and numbers them. It
also checks every letter. If an answer disagrees with a crossing word, the
engine refuses the answer and gives the reason. The model reads that reason and
tries again.

The model has four tools:

| Tool | Effect |
|---|---|
| `get_state` | Shows the grid, the clues, and the letters found so far. |
| `fill_slot` | Writes an answer into a slot. |
| `clear_slot` | Empties a slot. |
| `submit` | Ends the run. |

The run stops for one of three reasons. The model submits the grid. Or the
model stops calling tools. Or the agent reaches its turn limit.

The engine is the authority on the grid. The model is the authority on the
answers. This division is the core idea: a wrong answer stays a wrong answer,
and it cannot become a broken grid.

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

## Models

The agent speaks the OpenAI chat API, so two services serve it without any
change to the code. The dropdown on the page picks between them, and each
model is pinned to the service that runs it.

| Service | Models on the page | Key |
|---|---|---|
| **Nebius Token Factory** (default) | `deepseek-ai/DeepSeek-V4-Pro`, `Qwen/Qwen3-235B-A22B-Instruct-2507` | `NEBIUS_API_KEY` |
| Vercel AI Gateway | `deepseek/deepseek-v4-flash-0731` | `LLM_API_KEY`, or the Vercel OIDC token |

Nebius AI Studio is now called **Nebius Token Factory**; its host is
`https://api.tokenfactory.nebius.com/v1`, and you create a key at
<https://tokenfactory.nebius.com>. On Vercel the gateway needs no key at all,
because the deployment sends its own OIDC token.

Settings have this order of precedence: constructor arguments first, then the
`LLM_*` variables, then the `NEBIUS_*` variables. `api/index.py` passes the
host and the key for every request, so one request never inherits the other
service's settings.

### Picking a model on Nebius

**Check that a model supports tools before you use it.** The agent works
entirely through tool calls, so a model without them fails at once. The
catalog answers this directly:

```bash
curl -s -H "Authorization: Bearer $NEBIUS_API_KEY" \
  "https://api.tokenfactory.nebius.com/v1/models?verbose=true" |
  python3 -c "import json,sys; [print(m['id'], m.get('supported_features')) for m in json.load(sys.stdin)['data']]"
```

Of the 28 models in the catalog, 23 report `tools`. Note the trap:
`deepseek-ai/DeepSeek-V4-Flash` — the same family as the gateway default —
**does not**, so it cannot drive this agent on Nebius.

Measured on the 3×3 puzzle, the difference between models is large:

| Model | Result |
|---|---|
| `deepseek-ai/DeepSeek-V4-Pro` | Solved every puzzle. The default. |
| `Qwen/Qwen3-235B-A22B-Instruct-2507` | Solved the 3×3 in 2.5s, but failed the harder puzzles. |
| `nvidia/nemotron-3-super-120b-a12b` | Solved the two smaller puzzles only. |
| `meta-llama/Llama-3.3-70B-Instruct` | Solved the 3×3, but took 7 turns to do it. |
| `openai/gpt-oss-120b` | Failed the 3×3 at 33% of letters. |

The two Nebius models on the page are a deliberate pair: **DeepSeek V4 Pro for
dependable tool calling** — it solved every sample puzzle — and **Qwen3 235B
for raw speed**, answering the 3×3 in 2.5 seconds. Speed and tool use pull in
different directions on Nebius's catalog, and the pair covers both.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Then put your AI Gateway key or your Nebius key in `.env`.

## Run

Solve one puzzle from the command line. The output shows each tool call, the
final grid, and the score.

```bash
python scripts/solve.py data/puzzles/example_mini_5x5.json
```

Start the web page. Open <http://127.0.0.1:8000> after the server starts.

```bash
uvicorn api.index:app --reload
```

Score the agent on every puzzle in `data/puzzles/`. Three runs give you a
sense of the variation between attempts.

```bash
python -m eval.run_eval --solver llm --runs 3
```

## New puzzles

The demo can build a puzzle that did not exist before. Each half of the task
goes to the side that does it well.

**A search fills the grid.** `generator.py` selects a block layout. Then the
backtracking filler in `solver.py` fits words from `data/wordlist.txt` into it.
Every entry is therefore a real word, and every crossing agrees. The model
cannot produce a broken puzzle, because the model does not build the grid. A
5×5 grid fills in less than 0.1 seconds. The 7×7 grid takes about 1 second.

**The model writes the clues.** It returns the clues and a title in one JSON
reply. Each clue is then checked. If a clue contains its own answer, the code
replaces the answer with `___` and asks the model again. A blank is a fair
crossword clue, so nothing is lost.

**The agent solves blind.** It receives an empty grid and the clues. It does
not receive the answers. The answer key stays aside, and it is used only to
score the finished attempt.

Generation and solving are two separate requests: `POST /api/generate`, then
`POST /api/solve/stream` with the puzzle in the body. Together they take longer
than one function is allowed to run, so the browser holds the puzzle between
the two calls. The answer key travels with it, only so that the result can be
scored. The key is never part of what the agent reads.

### Speed is the weak point

Reasoning models think at length, and the time varies a great deal. On the
Vercel gateway, clue writing has taken between 52 and 206 seconds, and a blind
solve of a new 5×5 grid between 25 and 150 seconds. Nebius has been quicker in
testing: solves of the fixed puzzles ran 6 to 40 seconds.

Two measurements changed the design. The prompt used to tell the agent to call
`get_state` first. That instruction wasted a full turn, because the first
message already contains the grid. A second instruction told the agent to
confirm its work before it submitted. That check cost more than 200 seconds,
and it re-examined a grid the engine had already approved. Both instructions
are gone. A solve fell from about 420 seconds to 153 seconds.

Generation can still come close to the 300 second limit. If it does, the page
tells you, and you press the button again.

### One caveat

The same model writes the clues and then solves them. It cannot see the answers
while it solves. But its own choice of words may suit it better than a
stranger's would. Read the results for new puzzles as a demonstration. Read the
results for the fixed puzzles below as the measurement.

## Race the two services

The Race button answers one question: with the model held constant, which
service returns the answer sooner?

Press it, and the browser fires two identical requests at the same instant —
one to Nebius Token Factory, one to Vercel AI Gateway. Both run **DeepSeek V4
Pro**, the same weights on both sides, so the clock measures infrastructure
and not model quality. Two cards stream each agent's moves with their own
clocks. The verdict names the winner, gives both times and the ratio, and says
whether both sides solved the puzzle and produced the same grid.

Each race also adds a row to a **Speed comparison** chart on the page: one bar
for each service, drawn to a shared scale so races stay comparable, with the
times labeled on the bars.

The Generate button feeds the same race. It builds a puzzle that did not exist
before — grid by wordlist search, clues by the model — and then both services
solve it blind, simultaneously. Model parity holds there too: the same
DeepSeek V4 Pro weights on both sides, and a test pins the page's pair of
model ids to the server's.

Read the result with care. The clock starts in the browser, so it includes the
network and any cold start. One race is one sample: providers share load, and
turn counts differ between runs because sampling differs. Expect the ratio to
move between clicks. A race costs a few cents, because V4 Pro is a larger
model than the page's cheap default.

## Evaluation

Each metric is an average over the number of runs you request, because the
model does not answer the same way every time.

- **Letter accuracy.** Correct open cells, divided by total open cells. This
  metric gives partial credit.
- **Word accuracy.** Correct slots, divided by total slots. This metric
  punishes an answer that is almost right, because it breaks its crossings.
- **Solved rate.** The number of runs that produce a fully correct grid.
- **Cost.** The mean number of turns and tokens for each solve.

Four solvers bracket the score. Two of them also prove that the harness itself
is correct.

| Solver | What it does | Expected score |
|---|---|---|
| `empty` | Leaves the grid blank. | 0% — the floor |
| `backtrack` | Fits real words, but ignores the clues. | about 10% of letters |
| `llm` | The agent under test. | — |
| `oracle` | Copies the answer key. | 100% — the ceiling |

The distance between `backtrack` and `llm` is the part that clue understanding
contributes. `backtrack` shows what pure grid search achieves on its own.

The puzzle set has four hand-checked puzzles: a 3×3 square, two 5×5 puzzles,
and a 7×7 pinwheel. One 5×5 puzzle uses plain clues. The other uses wordplay.
Every puzzle carries an answer key, and the tests check the numbering, the clue
coverage, and the key.

The set is small, so read the numbers as an indication and not as proof. The
puzzles are English only. There are no themed puzzles and no rebus squares. The
harness also assumes that the answer key is the only correct solution.

## Results

**Nebius Token Factory, `deepseek-ai/DeepSeek-V4-Pro`** — two runs for each
puzzle, 10 August 2026. It solved all 8 runs.

| Puzzle | Letters | Words | Solved | Turns | Tokens | Time |
|---|---|---|---|---|---|---|
| example_3x3 | 100% | 100% | 2/2 | 2.0 | 3.2k | 5.9s |
| example_mini_5x5 | 100% | 100% | 2/2 | 10.5 | 48.8k | 39.1s |
| example_5x5_b | 100% | 100% | 2/2 | 5.0 | 14.7k | 19.0s |
| example_7x7 | 100% | 100% | 2/2 | 5.5 | 24.9k | 39.5s |

**Vercel AI Gateway, `deepseek/deepseek-v4-flash-0731`** — the same test, run
earlier. It also solved all 8 runs.

| Puzzle | Letters | Words | Solved | Turns | Tokens |
|---|---|---|---|---|---|
| example_3x3 | 100% | 100% | 2/2 | 3.5 | 5.9k |
| example_mini_5x5 | 100% | 100% | 2/2 | 3.5 | 15.1k |
| example_5x5_b | 100% | 100% | 2/2 | 3.5 | 9.8k |
| example_7x7 | 100% | 100% | 2/2 | 5.0 | 30.7k |

Both services solve every puzzle, so accuracy does not separate them here. The
puzzle set is too easy to rank two strong models, which is a limit of the set
and not a finding about the models.

The prompt was then shortened for speed. One further run of the same four
puzzles solved all of them again, and used fewer turns: 2, 3, 2 and 2. Those
runs took 18, 57, 123 and 85 seconds. The 7×7 puzzle fell from 5 turns to 2.

Both services are cheap for this workload. DeepSeek V4 Pro on Nebius costs
$1.75 for each million input tokens and $3.50 for output, so the eight runs
above used roughly 180k tokens and cost about 40 cents. The gateway model is
cheaper still, at $0.014 and $0.028 for each million.

Two other models ran on the earlier version of the loop.
`anthropic/claude-sonnet-4.5` solved all 8 of its runs. `openai/gpt-4o-mini`
failed its one run on the 3×3 puzzle, with 67% of letters and 33% of words. It
answered SIX for "Half a score", and it submitted the grid although the
crossings made no words. That failure is the reason for the word accuracy
metric and for the `backtrack` baseline.

## Tests

```bash
pytest
```

The suite has 54 tests and needs no API key. It covers the numbering, the
conflict checks, the puzzle files, the metrics, the graph, and the API.
The graph tests drive the agent with a scripted model, so the loop, the stop
rules, and the token counts are all tested offline.

## Deployment

The app runs on Vercel at <https://nebius-xword.vercel.app>. A push to `main`
starts a deployment. `vercel deploy --prod` also starts one.

Vercel serves `public/` directly. `vercel.json` sends the five `/api/*` routes
to the FastAPI function. Each rewrite carries the original path in a `__path`
parameter, because the Python runtime sees only the destination path.
Middleware in `api/index.py` puts the path back. `maxDuration` is 300 seconds,
which gives the agent room to think.

The gateway needs no key: middleware takes the Vercel OIDC token from the
request and uses it. Nebius needs one, so set `NEBIUS_API_KEY` as a project
variable:

```bash
vercel env add NEBIUS_API_KEY production
```

The public endpoints spend real credit, so `ALLOWED_MODELS` in `api/index.py`
limits them to the models in the dropdown. That limit guards only the hosted
demo. On your own machine, `LLM_MODEL`, `NEBIUS_MODEL` and `--model` still
accept any model either service offers.

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

`#` is a block. `.` is an open cell. The slot numbers come from the standard
crossword rules, so the keys in `clues` must agree with them.
`solver.verify_puzzle` checks this agreement for you. To build a new puzzle,
use `solver.fill_grid`, the same filler the generator uses.

## Next steps

- [x] Wordlist filler, for the baseline and for new grids
- [x] LangGraph agent loop
- [x] New puzzles, solved blind
- [ ] A `suggest(pattern)` tool, so the model can ask the wordlist for
      candidates when a clue defeats it
- [ ] Candidate lists for each clue, and a beam search to choose between them.
      This keeps the model out of the inner loop on large grids.
- [ ] A JSON protocol, for models that cannot call tools
- [ ] Importers for `.puz` files, to make the puzzle set larger
- [ ] A comparison of the models the gateway offers
