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

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from eval.metrics import score_grid  # noqa: E402
from nebius_xword.grid import Puzzle, load_puzzle  # noqa: E402

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
    if oidc and not os.getenv("LLM_API_KEY") and not os.getenv("NEBIUS_API_KEY"):
        os.environ["VERCEL_OIDC_TOKEN"] = oidc
    return await call_next(request)


def get_puzzle(puzzle_id: str) -> Puzzle:
    path = (PUZZLES_DIR / f"{puzzle_id}.json").resolve()
    if path.parent != PUZZLES_DIR.resolve() or not path.is_file():
        raise HTTPException(404, f"no such puzzle: {puzzle_id}")
    return load_puzzle(path)


# The public endpoint pays for gateway tokens, so only these models are allowed.
ALLOWED_MODELS = [
    "deepseek/deepseek-v4-flash-0731",
    "deepseek/deepseek-v4-flash",
]


class SolveRequest(BaseModel):
    puzzle_id: str
    solver: str = "llm"  # "llm" or "oracle" (demo without an API key)
    model: str | None = None


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
    puzzle = get_puzzle(puzzle_id)
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
    puzzle = get_puzzle(req.puzzle_id)
    validate_solve_request(req, puzzle)
    stats: dict = {"solver": req.solver}
    if req.solver == "oracle":
        grid = oracle_grid(puzzle)
    else:
        from nebius_xword.agent import CrosswordAgent  # deferred import: needs openai

        try:
            result = CrosswordAgent(model=req.model).solve(puzzle)
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
    puzzle = get_puzzle(req.puzzle_id)
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

            agent = CrosswordAgent(model=req.model)
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

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
