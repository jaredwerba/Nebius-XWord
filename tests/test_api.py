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


def test_default_model_is_allowed():
    from api.index import ALLOWED_MODELS, DEFAULT_MODEL

    assert DEFAULT_MODEL in ALLOWED_MODELS


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


def test_inline_generated_puzzle_can_be_solved():
    """A generated puzzle handed back by the browser solves and scores."""
    document = {
        "id": "generated-test",
        "title": "Round trip",
        "grid": ["...", "...", "..."],
        "clues": {"across": {"1": "a", "4": "b", "5": "c"},
                  "down": {"1": "d", "2": "e", "3": "f"}},
        "solution": ["CAT", "ARE", "TEN"],
    }
    res = client.post("/api/solve", json={"puzzle": document, "solver": "oracle"})
    assert res.status_code == 200
    assert res.json()["score"]["solved"] is True


def test_solve_requires_exactly_one_puzzle_source():
    assert client.post("/api/solve", json={"solver": "oracle"}).status_code == 400
    res = client.post(
        "/api/solve",
        json={"puzzle_id": "example_3x3", "puzzle": {"grid": ["..."]}, "solver": "oracle"},
    )
    assert res.status_code == 400


def test_malformed_inline_puzzle_rejected():
    res = client.post("/api/solve", json={"puzzle": {"nope": 1}, "solver": "oracle"})
    assert res.status_code == 400


@pytest.mark.parametrize("model", ["openai/o3-pro", "anthropic/claude-sonnet-4.5"])
def test_generate_rejects_disallowed_model(model):
    assert client.post("/api/generate", json={"model": model}).status_code == 400


@pytest.mark.parametrize(
    "model", ["openai/o3-pro", "anthropic/claude-sonnet-4.5", "openai/gpt-4o", "openai/gpt-4o-mini"]
)
def test_disallowed_model_rejected_without_network(model):
    res = client.post(
        "/api/solve", json={"puzzle_id": "example_3x3", "solver": "llm", "model": model}
    )
    assert res.status_code == 400


def test_allowlist_is_exactly_the_two_model_sets():
    from api.index import ALLOWED_MODELS, GATEWAY_MODELS, NEBIUS_MODELS

    assert set(ALLOWED_MODELS) == NEBIUS_MODELS | GATEWAY_MODELS
    assert not (NEBIUS_MODELS & GATEWAY_MODELS)  # a model belongs to one service


def test_each_model_routes_to_its_own_service(monkeypatch):
    """A Nebius id must reach Nebius, and a gateway id must not."""
    from api.index import GATEWAY_MODELS, NEBIUS_MODELS, connection_for
    from nebius_xword.agent import GATEWAY_BASE_URL, NEBIUS_BASE_URL

    monkeypatch.setenv("NEBIUS_API_KEY", "nebius-key")
    monkeypatch.setenv("VERCEL_OIDC_TOKEN", "gateway-token")
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    for model in NEBIUS_MODELS:
        conn = connection_for(model)
        assert conn["base_url"] == NEBIUS_BASE_URL
        assert conn["api_key"] == "nebius-key"

    for model in GATEWAY_MODELS:
        conn = connection_for(model)
        assert conn["base_url"] == GATEWAY_BASE_URL
        assert conn["api_key"] == "gateway-token"


def test_compare_pairs_are_consistent():
    """Each race pair runs one id on each service; the sets must agree."""
    from api.index import COMPARE_PAIRS, GATEWAY_MODELS, NEBIUS_MODELS

    for pair in COMPARE_PAIRS:
        assert pair["nebius"] in NEBIUS_MODELS
        assert pair["gateway"] in GATEWAY_MODELS
        assert pair["nebius"] != pair["gateway"]


def test_every_selectable_model_belongs_to_exactly_one_pair():
    """Whichever model the dropdown offers, a race must be able to pair it."""
    from api.index import ALLOWED_MODELS, COMPARE_PAIRS

    paired = [m for pair in COMPARE_PAIRS for m in (pair["nebius"], pair["gateway"])]
    assert sorted(paired) == sorted(ALLOWED_MODELS)  # no model without a twin
    assert len(paired) == len(set(paired))  # and none in two pairs


def test_page_and_server_agree_on_the_pairs():
    from pathlib import Path

    from api.index import COMPARE_PAIRS

    page = (Path(__file__).parents[1] / "public" / "index.html").read_text()
    for pair in COMPARE_PAIRS:
        assert pair["nebius"] in page
        assert pair["gateway"] in page
        assert pair["label"] in page


def test_nebius_base_url_has_no_trailing_slash():
    """A trailing slash would make the client build /v1//chat/completions."""
    from nebius_xword.agent import NEBIUS_BASE_URL

    assert NEBIUS_BASE_URL == "https://api.tokenfactory.nebius.com/v1"
