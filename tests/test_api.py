import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.index import app  # noqa: E402

client = TestClient(app)


def test_home_serves_ui():
    res = client.get("/")
    assert res.status_code == 200
    assert "Nebius-XWord" in res.text


def test_list_puzzles():
    res = client.get("/api/puzzles")
    assert res.status_code == 200
    ids = {p["id"] for p in res.json()}
    assert {"example_3x3", "example_mini_5x5", "example_5x5_b", "example_7x7"} <= ids


def test_puzzle_detail():
    res = client.get("/api/puzzles/example_7x7")
    assert res.status_code == 200
    body = res.json()
    assert len(body["grid"]) == 7
    assert len(body["slots"]) == 22
    assert all(slot["clue"] for slot in body["slots"])


def test_puzzle_detail_404():
    assert client.get("/api/puzzles/nope").status_code == 404
    assert client.get("/api/puzzles/..%2Fnope").status_code == 404


def test_oracle_solve_scores_perfect():
    res = client.post("/api/solve", json={"puzzle_id": "example_3x3", "solver": "oracle"})
    assert res.status_code == 200
    body = res.json()
    assert body["complete"] is True
    assert body["score"] == {"letter_accuracy": 1.0, "word_accuracy": 1.0, "solved": True}
    assert body["slots"]["1A"] == "CAT"


def test_unknown_solver_rejected():
    res = client.post("/api/solve", json={"puzzle_id": "example_3x3", "solver": "magic"})
    assert res.status_code == 400


def test_deepseek_default_in_allowlist():
    from api.index import ALLOWED_MODELS
    from nebius_xword.agent import GATEWAY_DEFAULT_MODEL

    assert GATEWAY_DEFAULT_MODEL == "deepseek/deepseek-v4-flash-0731"
    assert GATEWAY_DEFAULT_MODEL in ALLOWED_MODELS


def test_stream_oracle_emits_sse_and_done():
    import json

    with client.stream(
        "POST", "/api/solve/stream", json={"puzzle_id": "example_3x3", "solver": "oracle"}
    ) as res:
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        body = "".join(res.iter_text())
    events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    assert events[0]["event"] == "start"
    assert events[-1]["event"] == "done"
    assert events[-1]["score"]["solved"] is True
    assert events[-1]["grid"] == ["CAT", "ARE", "TEN"]


@pytest.mark.parametrize(
    "model", ["openai/o3-pro", "anthropic/claude-sonnet-4.5", "openai/gpt-4o", "openai/gpt-4o-mini"]
)
def test_disallowed_model_rejected_without_network(model):
    res = client.post(
        "/api/solve", json={"puzzle_id": "example_3x3", "solver": "llm", "model": model}
    )
    assert res.status_code == 400


def test_allowlist_is_deepseek_only():
    from api.index import ALLOWED_MODELS

    assert all(m.startswith("deepseek/") for m in ALLOWED_MODELS)
