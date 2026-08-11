# grok.md — machine-oriented technical documentation for Nebius-XWord

Audience: an AI session watching this folder. Human-facing prose lives in
README.md; this file trades readability for completeness and precision.
Everything below was measured or implemented in this repository; nothing is
aspirational. Dates are 2026-08-10/11 unless stated.

---

## 0. The take-home assignment, and exactly how it is answered

The assignment: **"Build an AI agent capable of solving crossword puzzles"**
with three deliverables.

### 0.1 "A working implementation of the agent"

- Implementation: `src/nebius_xword/` — a tool-calling LLM agent over a
  deterministic grid engine, orchestrated by LangGraph. Entry points:
  `CrosswordAgent.solve(puzzle) -> SolveResult` and
  `CrosswordAgent.stream(puzzle) -> Iterator[event dict]`
  (`src/nebius_xword/agent.py`).
- "Working" is proven, not claimed: the deployed instance
  (https://nebius-xword.vercel.app) has solved every fixed puzzle at 100%
  letter and word accuracy on live inference, on two independent providers
  (Nebius Token Factory, Vercel AI Gateway). See §7 for every measured run.
- Core invariant: **the engine is the authority on the grid; the model is
  the authority on answers only.** `Grid.fill_slot` rejects any word that
  conflicts with letters already placed, returning the conflicting cells.
  The model receives that rejection as a tool result and must revise. A
  hallucinated answer therefore cannot corrupt state; it can only fail.

### 0.2 "A proposed evaluation methodology for measuring the quality of its solutions"

- Harness: `eval/run_eval.py`; metrics: `eval/metrics.py::score_grid`.
- Metrics per puzzle, averaged over `--runs N` (LLM output is stochastic):
  - letter_accuracy = correct open cells / total open cells (partial credit)
  - word_accuracy = fully-correct slots / total slots (punishes near-misses,
    because a near-miss breaks crossings)
  - solved = all open cells correct (boolean; the headline)
  - cost = turns and total tokens per solve (from `usage_metadata`)
- Four solvers bracket the score and validate the harness itself:
  - `empty` → must score 0 (floor; proves the scorer)
  - `oracle` → copies the answer key, must score 100% (ceiling; proves the
    scorer end-to-end through the same code path)
  - `backtrack` → fills real words from a wordlist while ignoring clues;
    measured ≈9% letters. This is the structure-only baseline: the gap
    between `backtrack` and `llm` isolates clue understanding, which is the
    quantity the assignment actually asks about.
  - `llm` → the agent under test.
- Puzzle set: 4 hand-verified puzzles (3×3, 5×5 plain, 5×5 wordplay, 7×7
  pinwheel), each with an answer key; `solver.verify_puzzle` checks block
  symmetry-independent invariants: numbering, clue coverage, key consistency.
- Stated limitations (also in README): N is small; English-only; no
  themed/rebus; the key is assumed unique.

### 0.3 "A GitHub repository containing the source code and clear instructions"

- Public: https://github.com/jaredwerba/Nebius-XWord (main branch).
- README is written in ASD-STE100-style controlled English, first person,
  addressed to the reviewing hiring manager; setup is 4 commands; the test
  suite (65 tests) runs offline with no API key.

### 0.4 Extensions beyond the brief (each measurable on the live page)

1. Nebius brand system (logo asset, palette, Inter).
2. Streaming solve log with per-move timestamps, elapsed clock, ETA heuristic.
3. Puzzle generation: search-filled grids + LLM-written clues, solved blind.
4. Dual-provider support with strict model parity, plus a simultaneous race
   and an always-visible average-speed chart (Nebius 17.5s vs Vercel 53.7s
   mean over the 4 recorded races; 3.1×).
5. Honest failure documentation (see §8: Flash-no-tools trap, gpt-oss failure,
   the env-load race bug).
6. **External puzzles (§11)**: an importer for third-party crosswords that
   reads only rendered content, and a completed run against a real
   newspaper-size 13×13 daily — DeepSeek V4 Pro on Nebius filled all 60
   interlocking entries and submitted, in 98 turns / 996s / 2.44M tokens.
   This is the strongest single evidence that the agent generalizes past the
   repository's own fixtures.

---

## 1. Repository map with responsibilities

```
src/nebius_xword/
  grid.py       Grid/Slot/Puzzle models. Slot detection + standard numbering.
                fill_slot (conflict detection), clear_slot, set_rows,
                slot_pattern, render. puzzle_from_mapping/load_puzzle.
  tools.py      TOOL_SCHEMAS (OpenAI function format; single source of truth)
                and ToolExecutor: dispatches {get_state,fill_slot,clear_slot,
                submit} against a live Grid; JSON-encodes results; converts
                KeyError/ValueError/TypeError into {"error": ...} payloads.
  graph.py      LangGraph StateGraph. AgentState = MessagesState + turns:int
                + nudges:int. Nodes: agent_node (model.invoke over a windowed
                history), tools_node (executor), nudge_node (§11.4).
                Edges: START→agent; agent→(tools | nudge on prose | END);
                tools→(END if submitted or turns>=max_turns else agent);
                nudge→agent. window_messages bounds re-sent context.
  agent.py      resolve_llm_config (env precedence), build_chat_model
                (ChatOpenAI, use_responses_api=False), CrosswordAgent
                (solve/stream), SolveResult, SYSTEM_PROMPT.
  generator.py  TEMPLATES (5 layouts), fill_random_template (uses
                solver.fill_grid), write_clues (LLM JSON reply + validation +
                answer-leak masking + one retry), to_puzzle, puzzle_document.
  solver.py     matches/candidates (pattern ops), fill_grid (backtracking,
                §3), verify_puzzle (file validation).
  external.py   third-party import (§11): puzzle_from_scrape, check_import
                (numbering guard), audit_fill and compare_fills (scoring
                substitutes when no answer key exists), EXTRACTION_JS.
api/index.py    FastAPI app; also the Vercel serverless entry point. Routing
                of models to services (connection_for), SSE endpoints,
                middleware for Vercel quirks (§5.2, §5.3).
public/index.html  Entire frontend: no framework, no build step. SSE consumer,
                race orchestration, chart rendering.
eval/           run_eval.py (harness; solvers empty/oracle/backtrack/llm,
                --runs, --json), metrics.py (score_grid).
data/puzzles/   4 fixed puzzles, JSON (schema §2.3).
data/wordlist.txt  1,663 curated common words (lengths 3,4,5,7).
tests/          65 tests, all offline. Fake-model driven graph tests (§6).
scripts/solve.py  CLI single solve.
vercel.json     5 rewrites carrying __path (§5.2); maxDuration 300.
```

Dependency layering: `grid.py`/`solver.py`/`generator.py(minus write_clues)`
are stdlib-only. `agent.py` pulls langchain/langgraph/openai. `api/index.py`
defers the agent import inside request handlers so `GET /` and `/api/puzzles`
never pay the import cost.

## 2. The grid engine

### 2.1 Numbering

Standard crossword rules, implemented in `Grid._find_slots`: scan row-major;
a cell starts an across slot iff it is open, its left neighbor is a block or
edge, and the cell to its right is open; symmetric for down. A cell that
starts either direction gets the next integer. Slot id = f"{number}{'A'|'D'}".
Tests pin the numbering for the 5×5 (`1A,4A,5A,6A,7A / 1D..5D`) and the 7×7
(22 slots).

### 2.2 Mutation API

- `fill_slot(slot_id, word, overwrite=False) -> list[conflict strings]`.
  Length and charset validated (raises ValueError); conflicts leave the grid
  unchanged unless overwrite. Uppercases input.
- `clear_slot` blanks all cells of the slot (crossing letters are lost —
  documented behavior, and the agent prompt warns about it).
- `set_rows(rows)` bulk-loads a full solution (oracle path); block positions
  must match the template or ValueError.

### 2.3 Puzzle JSON schema

```json
{"id": str, "title": str,
 "grid": ["##...", ...],          // '#' block, '.' open; rectangular
 "clues": {"across": {"1": str}, "down": {...}},  // keys = slot numbers
 "solution": ["##PIT", ...] | absent}
```
`puzzle_from_mapping` builds a Puzzle from any decoded dict (used for inline
puzzles posted back by the browser); `load_puzzle` from a file.

## 3. The wordlist filler (`solver.fill_grid`)

Backtracking with two properties that matter:

1. **MRV ordering**: at each node, choose the unfilled slot with the fewest
   candidates; zero candidates → fail fast. Slots completed passively by
   crossings are validated against the wordset (a non-word passive completion
   fails the node). Seed slots (pre-filled before the call) are exempt.
2. **Position index**: words are indexed by (length, position, letter) into
   sets; candidates for a pattern = intersection of the sets for its fixed
   letters, minus used words, ordered by the rng-shuffled pool order. This
   replaced an O(pool) scan per slot per node and took the 7×7 pinwheel from
   effectively unfillable (>5 min, timeouts) to ~1s; 5×5s fill in <0.1s.

Known limitation (accepted): passive completions are checked for wordhood but
not for duplication; a duplicate can slip through rarely. The curated-quality
wordlist and template choice make this cosmetic for generation.

Uses: `backtrack` eval baseline; puzzle construction (the 5×5-b and 7×7 fixed
puzzles were built with it, then hand-clued); runtime generation (§4).

## 4. Generation pipeline (`generator.py` + `/api/generate`)

1. `fill_random_template`: pick template (5 verified layouts), run fill_grid
   with a fresh rng; retry across templates up to 12 attempts.
2. `write_clues(model, grid)`: one prompt lists slot ids + answers; model
   returns strict JSON `{"title", "clues": {slot_id: clue}}` (code fences
   stripped; first `{`..last `}` extracted). Validation: every slot clued;
   any clue containing its own answer (regex `\b{answer}\w*\b`, case-
   insensitive) is masked to `___` and counted as a problem; one corrective
   retry re-sends the problem list. Missing slots after retries → RuntimeError.
3. `to_puzzle` reassembles clues keyed by direction/number (the engine's
   numbering is the source of truth for keys).
4. The endpoint streams SSE phases (`grid`, `clues`), then a `puzzle` event
   that includes both the display view and `document` (full puzzle incl. key).
   Generation model: Nebius DeepSeek V4 Pro (`PAIR.nebius`).
5. The browser holds `document` and POSTs it back to `/api/solve/stream` as
   `puzzle` (inline) — twice, simultaneously: the generate button feeds the
   race (§5.5). Split rationale: generation+solve can exceed one 300s
   function budget; two requests each get their own.

Blindness property: the solver leg constructs a fresh Grid from the template
and clues; the key is carried only for post-hoc scoring in `solve_payload`.
The key does transit the browser (documented in README); it never enters a
model prompt.

Measured: grid fill <0.1s (5×5) / ~1s (7×7); clue writing on Nebius V4 Pro
3.0s, 46.8s, 51.7s observed; gateway-era (V4 Flash) 54–206s.

## 5. Service layer

### 5.1 Model→service routing

```python
NEBIUS_MODELS  = {"Qwen/Qwen3-235B-A22B-Instruct-2507", "deepseek-ai/DeepSeek-V4-Pro"}
GATEWAY_MODELS = {"deepseek/deepseek-v4-pro", "alibaba/qwen-3-235b"}
COMPARE_PAIR   = {"label": "DeepSeek V4 Pro",
                  "nebius": "deepseek-ai/DeepSeek-V4-Pro",
                  "gateway": "deepseek/deepseek-v4-pro"}
```
`connection_for(model)` returns explicit `{model, base_url, api_key}`;
Nebius ids → `https://api.tokenfactory.nebius.com/v1` + NEBIUS_API_KEY;
everything else → `https://ai-gateway.vercel.sh/v1` + (LLM_API_KEY or
VERCEL_OIDC_TOKEN). Explicit kwargs deliberately bypass the env fallback
chain in `resolve_llm_config` so a request can never inherit the other
service's credential. The gateway sets mirror the Nebius models exactly
(same weights, per-service ids): verified `alibaba/qwen-3-235b` ==
Qwen3-235B-A22B-Instruct-2507 via the gateway catalog; tools=true on all
serving providers for all four ids.

No trailing slash on the Nebius base URL: `/v1/` + client-appended path
yields `/v1//chat/completions` → Nebius 404s.

Prices (verified from both live catalogs): Nebius V4 Pro $1.75/$3.50 per M
in/out; gateway V4 Pro $1.74/$3.48; Nebius Qwen $0.20/$0.60; gateway Qwen
$0.22/$0.88. Nebius rebrand note: "AI Studio" → "Token Factory"; old host
api.studio.nebius.com still answers as an alias.

### 5.2 Vercel Python runtime quirks (hard-won)

- Rewrites replace the path: the function sees the rewrite destination
  (`/api/index`), not the original URL. No original-path header exists.
  Fix: each rewrite in vercel.json carries `?__path=/api/...`; ASGI
  middleware rewrites `request.scope["path"]` before routing.
- Dependency resolution: with a root pyproject.toml, Vercel installs the
  project's own dependencies and ignores requirements.txt placed next to the
  entry point. FastAPI therefore lives in `[project.dependencies]`.
- Static: `public/` is served by the CDN ahead of rewrites; the FastAPI app
  additionally serves `/`, the logo, and the résumé for local-dev parity.
- SSE works through `StreamingResponse` (+`X-Accel-Buffering: no`).

### 5.3 Auth

- Gateway: Vercel injects the OIDC token per-request as header
  `x-vercel-oidc-token` (not as an env var at runtime). Middleware exports it
  to `os.environ["VERCEL_OIDC_TOKEN"]` (unconditionally unless LLM_API_KEY is
  set — see §8 bug 3). The gateway accepts it as a Bearer token; zero
  configured secrets for that path.
- Nebius: NEBIUS_API_KEY as a Vercel project env var and in local `.env`
  (gitignored; `.vercelignore` additionally keeps .env/.venv out of CLI
  deploy bundles).
- `load_dotenv(ROOT/".env")` at api module import — see §8 bug 2 for why.

### 5.4 SSE protocol (both solve streams and generation)

Event dicts, one per `data:` line:
`{"event":"start"|"llm"|"tool_call"|"tool_result"|"phase"|"puzzle"|"done"|"error", ...}`.
`agent.stream()` yields start/llm/tool_call/tool_result and a terminal
`{"event":"result","result":SolveResult}` which the API converts to `done`
(grid rows, per-slot words, score-if-key, stats {model,turns,tokens,
submitted}). `solve()` is implemented as "drain stream, return the result" —
one code path.

### 5.5 The race

Browser fires both `/api/solve/stream` POSTs in the same tick
(`Promise.allSettled`), one per COMPARE_PAIR id; each is an independent
Vercel invocation with its own 300s budget. Shared t0; per-card clocks; a
failed leg reports "did not finish" while the other completes. Verdict
compares client-measured wall times, reports ratio, both scores, and grid
equality. Every completed race appends to an in-session chart (two bars,
shared scale, longest bar capped at 74% width so value labels never clip);
a static "recorded average" block (17.5s vs 53.7s, 3.1×, n=4, all DeepSeek
V4 Pro) is hardcoded in the HTML with provenance in a comment.

## 6. Testing strategy (65 tests, all offline)

- Grid: numbering, lengths, conflicts, validation, clear, set_rows.
- Puzzles: every data/puzzles/*.json passes verify_puzzle (glob-based).
- Metrics: oracle=1.0, empty=0.0, partial fractions exact.
- Generator: template fillability (all 5), seed reproducibility, clue
  parsing incl. fenced JSON, answer-leak masking (incl. plural leak),
  missing-clue RuntimeError, blind-view property (fresh grid empty, clues
  present), document round-trip.
- Graph/agent: `GenericFakeChatModel` (langchain_core) scripted with
  AIMessages carrying tool_calls + usage_metadata. It raises StopIteration
  if the loop over-asks — a free over-loop guard. Because fake models do not
  implement bind_tools, `build_solver_graph` takes a **pre-bound** model;
  CrosswordAgent has a `chat_model=` injection seam. Covered: submit stop,
  no-tool-call stop, max_turns cap (scripted 4th turn would explode),
  usage=None tolerance, error-result continuation, parallel tool calls with
  submit, stream event sequence, token sums.
- API: TestClient; oracle solve, SSE oracle stream parse, inline-puzzle
  solve, exactly-one-of puzzle_id/puzzle, malformed inline 400, allowlist
  rejections per removed model, COMPARE_PAIR consistency with the model
  sets, page↔server id agreement (reads index.html), Nebius base URL shape.

## 7. Measured results (all on record in this repo's history)

Fixed-set evals (2 runs/puzzle unless noted):

| Config | 3×3 | 5×5 mini | 5×5-b | 7×7 | Solved |
|---|---|---|---|---|---|
| Nebius V4 Pro (LangGraph) | 2.0t/3.2k/5.9s | 10.5t/48.8k/39.1s | 5.0t/14.7k/19.0s | 5.5t/24.9k/39.5s | 8/8 |
| Gateway V4 Flash (historical) | 3.5t/5.9k | 3.5t/15.1k | 3.5t/9.8k | 5.0t/30.7k | 8/8 |
| Gateway Sonnet 4.5 (pre-port) | — | — | — | — | 8/8 |
| Gateway gpt-4o-mini (1 run) | 67% letters | — | — | — | 0/1 |

Nebius model bake-off (single runs): V4 Pro solved everything tried; Qwen3
235B 3×3 in 2.5s but 1/4 on the full set (74%/37%/83% letters on the
harder three); nemotron-3-super 2/4; Llama-3.3-70B solved 3×3 in 7 turns;
gpt-oss-120b failed 3×3 (33% letters). GLM-5.1 solved 5×5-b (2 turns, 54.6s).

Races (same weights both sides, wall-clock from client):
8.9 vs 25.0 · 11.8 vs 24.8 · 19.5 vs 77.8 (prod) · 29.8 vs 87.3 (generated,
blind). Means 17.5 vs 53.7 (3.1×). Prompt-trim effect: removing "call
get_state first" and "re-verify before submit" cut a generated-5×5 solve
from ~420s to 153s; post-trim regression run solved 4/4 in 2/3/2/2 turns
(18/57/123/85s).

## 8. Bugs found and fixed (worth knowing before touching anything)

1. **fill_grid passive completions** (fixed): crossings could complete a slot
   to a non-word silently; now validated at every node.
2. **Env-load race** (fixed): the first request in a fresh process called
   `connection_for` before anything had run `load_dotenv`; api_key resolved
   None, and `resolve_llm_config`'s fallback then substituted the *Nebius*
   key into a *gateway* request → gateway 401. Only surfaced when the race
   fired both services concurrently on a cold server. Fix: load_dotenv at
   module import; regression covered by routing tests with monkeypatched env.
3. **OIDC starvation guard** (fixed): middleware originally skipped exporting
   the OIDC token when NEBIUS_API_KEY existed — which would have broken all
   gateway models the moment the Nebius key was configured. Now exported
   unless LLM_API_KEY overrides.
4. **Nebius V4 Flash has no `tools`** (avoided): catalog `supported_features`
   lacks "tools" for deepseek-ai/DeepSeek-V4-Flash; it cannot drive this
   agent. Always check `GET /v1/models?verbose=true` before adopting a model.
5. **Vercel path rewriting / pyproject-vs-requirements / trailing slash**: §5.2.

## 9. Operational notes

- The résumé PDF (`public/jared-werba-resume.pdf`) is deliberately untracked
  (removed from git history by a force-push rewrite). Consequence: a deploy
  triggered by `git push` lacks it (page hides the link, endpoint 404s); a
  CLI `vercel deploy --prod --yes` from the working tree restores it. The
  routine after every push to main is therefore push → wait → CLI deploy.
- `ALLOWED_MODELS` guards the public endpoints because they spend real
  credit (Nebius prepaid; gateway OIDC). Local CLI/eval accept any id.
- The NEBIUS_API_KEY appeared in this conversation's transcript; the owner
  accepted the exposure and declined rotation for now.
- Local OIDC tokens (vercel env pull) expire after ~12h; symptoms: gateway
  legs 401 locally while Nebius legs work. Re-pull to a scratch file and
  merge — do not clobber NEBIUS_API_KEY in .env.
- Nebius rate limits are dynamic per-model (headers x-ratelimit-*); none hit
  during this work.

## 10. Branches

- `main`: everything in this document. HEAD at time of writing: c3de98b.
- `feature/puzzle-generator`: historical; merged (fast-forward).
- `external`: the third-party importer and the 160-turn work below; merged
  into main (fast-forward) on 2026-08-11. Kept as a marker, identical to main.

## 11. External puzzles — method, guardrail, and final results

### 11.1 Import method

`src/nebius_xword/external.py` + `tests/test_external.py` (5 tests, synthetic
fixtures only). Target: the boatloadpuzzles.com daily 13×13, 60 slots.

The importer consumes a scrape of **what the page renders**:
- block layout from `.grect` / `.gblacksquare` divs, positioned on a 25px
  lattice (13 unique x, 13 unique y → 13×13);
- clue text from `td.cnum` (number) and `td.cfullclue` (text), DOM order,
  across/down split at the first number decrease;
- `EXTRACTION_JS` in the module is the exact browser snippet used.

`check_import` is the load-bearing guard: the engine derives slot numbers
independently from the scraped blocks, and the import raises unless that set
equals the page's printed clue numbers. On the real target: 60/60 matched,
which is what makes the scrape trustworthy enough to solve against.

### 11.2 The guardrail (deliberate, not a gap)

The publisher ships the solution as an encrypted blob (`puzBody`, served by
`/getcrossword`, decoded by their minified `Crossword.js`). This project does
**not** request, decode, or reverse that blob. Consequences that propagate
through the code:
- imported `Puzzle.solution is None`; `score_grid` is therefore unavailable;
- substitute signals live in the same module: `audit_fill` (completeness,
  engine-verified crossing consistency, share of entries present in a
  reference dictionary, list of unknowns) and `compare_fills` (per-slot
  agreement between two independent solves);
- puzzle text never enters git: `data/external/` is gitignored. The repo
  ships code, not the publisher's content.

### 11.3 Final measured results (2026-08-11, all on Nebius Token Factory)

Round 1 — max_turns=40, full history (pre-window):

| Model | Wall | Turns | Tokens | Filled | Note |
|---|---|---|---|---|---|
| Qwen3-235B | 42.7s | 40 (cap) | 234k | 16/60 | thrash: fill→conflict→clear |
| DeepSeek V4 Pro | 282.2s | 40 (cap) | 1.09M | 57/60 | 13 out-of-dictionary |

Agreement 5/60. Cost ≈ $2.40. Diagnosis: quadratic history cost plus too small
a turn budget, not (only) model capability.

Round 2 — max_turns=160, history_window=48 (cost linear in turns):

| Model | Wall | Turns | Tokens | Cost | Result |
|---|---|---|---|---|---|
| **DeepSeek V4 Pro** | **996.0s** | **98** | **2.44M** | **≈$4.40** | **60/60, submitted, complete** |
| Qwen3-235B (nudged) | 146.4s | 160 (cap) | 787k | ≈$0.16 | 36/60, incomplete |
| Qwen3-235B (first try) | 7.1s | 1 | 2.8k | ≈$0.00 | 0/60 — prose exit, see §11.4 |
| GLM-5.1 | — | — | — | — | stopped by owner; ~27s/turn ⇒ ~1h projected |

- V4 Pro is the headline: it **chose** to call `submit` (not a cap stop) with
  every one of 60 interlocking entries filled and every crossing verified by
  the engine. 80% of its entries appear in `/usr/share/dict/words`; several
  "unknowns" are plausibly abbreviations the dictionary lacks — the puzzle has
  explicit "(abbr.)" clues.
- Cross-model agreement V4 Pro vs Qwen: 12/60 (20%). Qwen filled only 36
  slots, so this is weak corroboration and is reported as such.
- GLM-5.1 was launched as a same-strength second opinion from a different
  model family (the point being that agreement between families is
  evidentially stronger than within one). It was stopped mid-run on owner
  instruction: the projected ~1h runtime is unusable for a hiring-manager
  demo, and V4 Pro's completion was deemed sufficient.
- Total external-experiment spend ≈ $7.5 of the $50 credit.

**Claim discipline.** What is demonstrated: the agent can *complete* a real
newspaper-size crossword with full structural consistency, unaided, in ~17
minutes. What is NOT demonstrated: per-answer correctness, which cannot be
computed without the publisher's key, and which 20% cross-model agreement
does not establish. Both the page and the README state exactly this.

### 11.4 Two agent improvements this run forced

1. **History window** (`graph.window_messages`, default 48). Every turn used
   to resend the entire conversation → O(turns²) tokens; one 40-turn solve
   cost 1.09M. The window keeps `messages[:2]` (system prompt + initial grid
   and clues) plus the last N, and drops any leading `ToolMessage` after the
   cut so the API never sees a tool result without its originating call. Cost
   is now linear; an unwindowed 160-turn V4 Pro run was projected at ~$35.
   The model re-syncs anything scrolled out by calling `get_state`.
2. **Nudge node** (`graph.py`, `AgentState.nudges`). A model that answers in
   prose used to end the run outright, because "no tool calls" meant done —
   this killed the first 160-turn Qwen attempt at turn 1 with 0/60. The graph
   now routes a prose reply to a `nudge` node that appends a corrective user
   message and returns to the agent, at most 3 times, so a model that never
   calls tools still terminates. `recursion_limit` raised to
   `2*max_turns + 12` to accommodate the extra hops.

Both are covered by offline tests (window head/tail/orphan behavior; a
prose-only model terminating after the nudge budget; a prose start that
recovers and solves).

### 11.5 Presentation

The live page carries a recorded "A real external puzzle" section (not a live
run — a 13×13 takes ~17 min, which no reviewer will wait through), with the
V4 Pro numbers, the Qwen shortfall, the GLM pacing decision, and the
encryption guardrail stated plainly. README mirrors it.
