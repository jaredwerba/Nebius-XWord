"""FastAPI layer for Nebius-XWord — thin wrapper over the core library.

Deployed on Vercel (see vercel.json: every path rewrites to this ASGI app).
Run locally with:  uvicorn api.index:app --reload
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import json  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

# Load local credentials before any request handler reads the environment.
# Without this, the first request resolves keys before .env is loaded, and the
# fallback chain can hand one service the other service's key. On Vercel the
# file does not exist and the platform provides the variables, so this no-ops.
load_dotenv(ROOT / ".env")

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from eval.metrics import score_grid  # noqa: E402
from nebius_xword.grid import Puzzle, load_puzzle, puzzle_from_mapping  # noqa: E402

PUZZLES_DIR = ROOT / "data" / "puzzles"

app = FastAPI(title="Nebius-XWord", version="0.1.0")


@app.middleware("http")
async def vercel_bridge(request: Request, call_next):
    """Adapt Vercel's proxy quirks.

    Rewrites replace the request path with the function path, so vercel.json
    forwards the original path in ``__path`` and we restore it here. Vercel
    also delivers the OIDC token as a header; expose it as an env var so the
    agent can use it for AI Gateway auth without a configured key.
    """
    original = request.query_params.get("__path")
    if original and original.startswith("/api/"):
        request.scope["path"] = original
        request.scope["raw_path"] = original.encode()
    oidc = request.headers.get("x-vercel-oidc-token")
    if oidc and not os.getenv("LLM_API_KEY"):
        # Always export it. A Nebius key must not starve the gateway models of
        # their credential, because each request picks its own backend below.
        os.environ["VERCEL_OIDC_TOKEN"] = oidc
    return await call_next(request)


def get_puzzle(puzzle_id: str) -> Puzzle:
    path = (PUZZLES_DIR / f"{puzzle_id}.json").resolve()
    if path.parent != PUZZLES_DIR.resolve() or not path.is_file():
        raise HTTPException(404, f"no such puzzle: {puzzle_id}")
    return load_puzzle(path)


# The public endpoints spend real credit, so only these models may be selected.
# The service that serves each model is implied by which set it belongs to.
NEBIUS_MODELS = {
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "deepseek-ai/DeepSeek-V4-Pro",
}
# The gateway side mirrors the Nebius models exactly — the same weights under
# each service's own id — so every cross-service comparison is apples to apples.
GATEWAY_MODELS = {
    "deepseek/deepseek-v4-pro",
    "alibaba/qwen-3-235b",  # Qwen3-235B-A22B-Instruct-2507, same as the Nebius id
}
ALLOWED_MODELS = sorted(NEBIUS_MODELS | GATEWAY_MODELS)
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Pro"

# The race on the page runs the SAME weights on both services, so the times
# compare infrastructure and not model quality. The page hardcodes these two
# ids; tests keep it consistent with the sets above.
COMPARE_PAIR = {
    "label": "DeepSeek V4 Pro",
    "nebius": "deepseek-ai/DeepSeek-V4-Pro",
    "gateway": "deepseek/deepseek-v4-pro",
}


def connection_for(model: str | None) -> dict:
    """Pin a model to the service that serves it.

    Both services speak the OpenAI chat API, so the only difference is the host
    and the credential. Passing them explicitly keeps one request from
    inheriting another service's settings from the environment.
    """
    from nebius_xword.agent import GATEWAY_BASE_URL, NEBIUS_BASE_URL  # deferred

    model = model or DEFAULT_MODEL
    if model in NEBIUS_MODELS:
        return {"model": model, "base_url": NEBIUS_BASE_URL,
                "api_key": os.getenv("NEBIUS_API_KEY")}
    return {"model": model, "base_url": GATEWAY_BASE_URL,
            "api_key": os.getenv("LLM_API_KEY") or os.getenv("VERCEL_OIDC_TOKEN")}


class SolveRequest(BaseModel):
    puzzle_id: str | None = None  # a bundled puzzle...
    puzzle: dict | None = None  # ...or a generated one handed back by the browser
    solver: str = "llm"  # "llm" or "oracle" (demo without an API key)
    model: str | None = None


class GenerateRequest(BaseModel):
    model: str | None = None
    template: str | None = None


def puzzle_view(puzzle) -> dict:
    """The solver's-eye view of a puzzle: layout and clues, never the answers."""
    grid = puzzle.make_grid()
    return {
        "id": puzzle.id,
        "title": puzzle.title,
        "grid": puzzle.grid_rows,
        "has_solution": puzzle.solution is not None,
        "slots": [
            {
                "id": slot.id,
                "number": slot.number,
                "direction": slot.direction,
                "row": slot.row,
                "col": slot.col,
                "length": slot.length,
                "clue": slot.clue,
            }
            for slot in grid.slots.values()
        ],
    }


@app.get("/")
def home() -> FileResponse:
    return FileResponse(ROOT / "public" / "index.html")


@app.get("/nebius-logo.svg")
def logo() -> FileResponse:  # Vercel serves public/ statically; these are for local dev
    return FileResponse(ROOT / "public" / "nebius-logo.svg", media_type="image/svg+xml")


@app.get("/jared-werba-resume.pdf")
def resume() -> FileResponse:
    # Untracked by git (see .gitignore): present in CLI deploys, absent in
    # deploys built from the repo. The page hides the link when it is missing.
    path = ROOT / "public" / "jared-werba-resume.pdf"
    if not path.is_file():
        raise HTTPException(404, "resume not included in this deployment")
    return FileResponse(path, media_type="application/pdf")


@app.get("/jared-werba-pitch.mp4")
def pitch_video() -> FileResponse:
    # Same pattern as the resume: untracked, carried by CLI deploys only.
    path = ROOT / "public" / "jared-werba-pitch.mp4"
    if not path.is_file():
        raise HTTPException(404, "video not included in this deployment")
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/puzzles")
def list_puzzles() -> list[dict]:
    out = []
    for path in sorted(PUZZLES_DIR.glob("*.json")):
        puzzle = load_puzzle(path)
        out.append(
            {
                "id": puzzle.id,
                "title": puzzle.title,
                "rows": len(puzzle.grid_rows),
                "cols": len(puzzle.grid_rows[0]),
                "has_solution": puzzle.solution is not None,
            }
        )
    return out


@app.get("/api/puzzles/{puzzle_id}")
def puzzle_detail(puzzle_id: str) -> dict:
    return puzzle_view(get_puzzle(puzzle_id))


def resolve_puzzle(req: SolveRequest):
    """A bundled puzzle by id, or one posted back after generation.

    Either way the solver is handed only the layout and clues — the answer
    key rides along solely so the finished grid can be scored.
    """
    if (req.puzzle_id is None) == (req.puzzle is None):
        raise HTTPException(400, "provide exactly one of puzzle_id or puzzle")
    if req.puzzle_id is not None:
        return get_puzzle(req.puzzle_id)
    try:
        return puzzle_from_mapping(req.puzzle)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"malformed puzzle: {exc}") from exc


def validate_solve_request(req: SolveRequest, puzzle) -> None:
    if req.solver not in ("oracle", "llm"):
        raise HTTPException(400, f"unknown solver: {req.solver}")
    if req.solver == "oracle" and puzzle.solution is None:
        raise HTTPException(400, "puzzle has no answer key")
    if req.solver == "llm" and req.model is not None and req.model not in ALLOWED_MODELS:
        raise HTTPException(400, f"model must be one of {ALLOWED_MODELS}")


def oracle_grid(puzzle):
    grid = puzzle.make_grid()
    grid.set_rows(puzzle.solution)
    return grid


def solve_payload(grid, puzzle, stats: dict) -> dict:
    return {
        "grid": grid.render().split("\n"),
        "slots": {slot_id: grid.slot_pattern(slot_id) for slot_id in grid.slots},
        "complete": grid.is_complete(),
        "score": score_grid(grid, puzzle.solution) if puzzle.solution else None,
        "stats": stats,
    }


@app.post("/api/solve")
def solve(req: SolveRequest) -> dict:
    puzzle = resolve_puzzle(req)
    validate_solve_request(req, puzzle)
    stats: dict = {"solver": req.solver}
    if req.solver == "oracle":
        grid = oracle_grid(puzzle)
    else:
        from nebius_xword.agent import CrosswordAgent  # deferred import: needs openai

        try:
            result = CrosswordAgent(**connection_for(req.model)).solve(puzzle)
        except Exception as exc:  # surface config/provider errors to the UI
            raise HTTPException(502, f"LLM solve failed: {exc}") from exc
        grid = result.grid
        stats.update(
            model=result.model,
            turns=result.turns,
            tokens=result.total_tokens,
            submitted=result.submitted,
        )
    return solve_payload(grid, puzzle, stats)


@app.post("/api/solve/stream")
def solve_stream(req: SolveRequest) -> StreamingResponse:
    """Server-sent events: progress log lines while the agent solves."""
    puzzle = resolve_puzzle(req)
    validate_solve_request(req, puzzle)

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    def events():
        try:
            if req.solver == "oracle":
                yield sse({"event": "start", "model": "oracle", "puzzle": puzzle.id})
                grid = oracle_grid(puzzle)
                yield sse({"event": "tool_result", "tool": "oracle",
                           "result": "grid filled from the answer key"})
                yield sse({"event": "done",
                           **solve_payload(grid, puzzle, {"solver": "oracle"})})
                return
            from nebius_xword.agent import CrosswordAgent  # deferred import

            agent = CrosswordAgent(**connection_for(req.model))
            for event in agent.stream(puzzle):
                if event["event"] == "result":
                    result = event["result"]
                    stats = {"solver": "llm", "model": result.model, "turns": result.turns,
                             "tokens": result.total_tokens, "submitted": result.submitted}
                    yield sse({"event": "done",
                               **solve_payload(result.grid, puzzle, stats)})
                else:
                    yield sse(event)
        except Exception as exc:  # surface config/provider errors into the log
            yield sse({"event": "error", "message": f"LLM solve failed: {exc}"})

    return sse_response(events())


def sse_response(generator) -> StreamingResponse:
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/generate")
def generate(req: GenerateRequest) -> StreamingResponse:
    """Build a brand-new puzzle and stream progress while doing it.

    The grid is filled by wordlist search, so every entry is a real word and
    every crossing is consistent — the model cannot produce a broken puzzle.
    The model writes the clues. Solving is a separate request, so neither call
    has to fit generation and solving into one function timeout.
    """
    if req.model is not None and req.model not in ALLOWED_MODELS:
        raise HTTPException(400, f"model must be one of {ALLOWED_MODELS}")

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    def events():
        try:
            import random

            from nebius_xword.agent import build_chat_model  # deferred: heavy import
            from nebius_xword.generator import (
                fill_random_template,
                puzzle_document,
                to_puzzle,
                write_clues,
            )

            yield sse({"event": "phase", "phase": "grid",
                       "message": "searching the wordlist for a consistent grid…"})
            name, grid = fill_random_template(rng=random.Random(), template=req.template)
            yield sse({"event": "phase", "phase": "clues",
                       "message": f"{name} grid filled with real words — the model is now "
                                  "writing the clues (the slow step, up to ~90s)…"})

            title, clues = write_clues(build_chat_model(**connection_for(req.model)), grid)
            puzzle = to_puzzle(name, grid, title, clues)
            yield sse({"event": "puzzle", **puzzle_view(puzzle),
                       "document": puzzle_document(puzzle)})
        except Exception as exc:  # surface generation/provider errors into the log
            yield sse({"event": "error", "message": f"generation failed: {exc}"})

    return sse_response(events())
