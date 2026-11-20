"""Anonymous logging of what was shown and what visitors did with it."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as main_mod
from app.main import app, set_result_key
from app.models import Impression, Interaction

FILMS = [
    {"imdb_id": f"tt000000{i}", "title": f"Film {i}", "overview": "p", "keywords": "k"}
    for i in range(1, 4)
]
IDS = [f["imdb_id"] for f in FILMS]
MOOD = "folk horror"
KEY = set_result_key(MOOD, IDS)


@pytest.fixture(autouse=True)
def _ranker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main_mod, "recommend_movies_advanced", lambda **kw: [dict(f) for f in FILMS]
    )


def _search(client: Any, mood: str = MOOD) -> str:
    resp = client.get("/recommend", params={"mood": mood, "strategy": "semantic"})
    assert resp.status_code == 200
    return str(resp.text)


def _act(client: Any, imdb_id: str = "tt0000002", kind: str = "open", key: str = KEY) -> Any:
    return client.post(
        "/api/interaction", json={"result_key": key, "imdb_id": imdb_id, "kind": kind}
    )


class TestImpressions:
    def test_a_search_records_what_was_shown(self, client: Any, db: Any) -> None:
        page = _search(client)
        row = db.query(Impression).one()
        assert (row.mood, row.strategy, row.imdb_ids, row.result_key) == (
            MOOD,
            "semantic",
            IDS,
            KEY,
        )
        assert f'var RESULT_KEY = "{KEY}"' in page

    def test_reloading_the_same_set_is_one_impression(self, client: Any, db: Any) -> None:
        _search(client)
        _search(client)
        assert db.query(Impression).count() == 1

    def test_no_account_or_address_is_stored(self, client: Any, db: Any) -> None:
        _search(client)
        columns = {c.name for c in Impression.__table__.columns}
        assert "user_id" not in columns and not any("ip" in c for c in columns)
        assert len(db.query(Impression).one().visitor_id) == 32

    def test_empty_results_record_nothing(
        self, client: Any, db: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(main_mod, "recommend_movies_advanced", lambda **kw: [])
        assert "var RESULT_KEY = null" in _search(client)
        assert db.query(Impression).count() == 0

    def test_a_shared_set_is_not_measured(
        self, client: Any, db: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import app.services.corpus as corpus_mod

        monkeypatch.setattr(corpus_mod, "get_corpus_and_embeddings", lambda: (FILMS, None))
        page = client.get("/recommend", params={"mood": MOOD, "ids": ",".join(IDS)}).text
        assert "var RESULT_KEY = null" in page
        assert db.query(Impression).count() == 0

    def test_a_failure_to_record_never_breaks_the_search(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*args: Any, **kwargs: Any) -> str:
            raise RuntimeError("database is having a day")

        monkeypatch.setattr(main_mod, "visitor_id_for", _boom)
        page = _search(client)
        assert "Film 1" in page


class TestInteractions:
    def test_recorded_with_the_rank_the_server_knows(self, client: Any, db: Any) -> None:
        _search(client)
        assert _act(client, "tt0000003", "imdb_click").json() == {"status": "recorded"}
        row = db.query(Interaction).one()
        assert (row.imdb_id, row.kind, row.position, row.result_key) == (
            "tt0000003",
            "imdb_click",
            2,
            KEY,
        )

    def test_the_same_action_twice_is_one_signal(self, client: Any, db: Any) -> None:
        _search(client)
        assert _act(client).json()["status"] == "recorded"
        assert _act(client).json()["status"] == "duplicate"
        assert _act(client, kind="save").json()["status"] == "recorded"
        assert db.query(Interaction).count() == 2

    def test_ignored_for_a_set_this_visitor_was_never_shown(self, client: Any, db: Any) -> None:
        assert _act(client).json() == {"status": "ignored"}
        _search(client)
        assert _act(client, key="0" * 64).json() == {"status": "ignored"}
        with TestClient(app) as other:  # a different visitor replaying the key
            token = re.search(r'name="csrf-token" content="([^"]+)"', other.get("/").text)
            assert token
            body = {"result_key": KEY, "imdb_id": "tt0000002", "kind": "open"}
            replay = other.post("/api/interaction", json=body, headers={"X-CSRF-Token": token[1]})
            assert replay.json() == {"status": "ignored"}
        assert db.query(Interaction).count() == 0

    def test_ignored_for_a_film_that_was_not_in_the_set(self, client: Any, db: Any) -> None:
        _search(client)
        assert _act(client, imdb_id="tt9999999").json() == {"status": "ignored"}
        assert db.query(Interaction).count() == 0

    @pytest.mark.parametrize(
        "body",
        [
            {"result_key": "short", "imdb_id": "tt0000001", "kind": "open"},
            {"result_key": KEY, "imdb_id": "tt0000001", "kind": "stare"},
            {"result_key": KEY, "imdb_id": "", "kind": "open"},
            {"result_key": KEY.upper(), "imdb_id": "tt0000001", "kind": "open"},
        ],
    )
    def test_malformed_bodies_are_422(self, client: Any, body: dict[str, str]) -> None:
        assert client.post("/api/interaction", json=body).status_code == 422

    def test_requires_the_csrf_token(self, client: Any) -> None:
        body = {"result_key": KEY, "imdb_id": "tt0000001", "kind": "open"}
        resp = client.post("/api/interaction", json=body, headers={"X-CSRF-Token": "wrong"})
        assert resp.status_code == 403


class TestPageHooks:
    def test_every_measured_action_calls_the_logger(self, client: Any) -> None:
        page = _search(client)
        assert "function logInteraction" in page
        for kind in ("open", "imdb_click", "save", "similar_click"):
            assert f"'{kind}')" in page, f"no hook logs {kind!r}"
        assert "keepalive: true" in page  # an IMDb click navigates away


def _export_module() -> Any:
    path = Path(__file__).resolve().parent.parent / "scripts" / "export_eval_set.py"
    spec = importlib.util.spec_from_file_location("export_eval_set", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestEvalExport:
    @pytest.fixture(autouse=True)
    def _corpus(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import app.services.corpus as corpus_mod

        monkeypatch.setattr(corpus_mod, "load_corpus", lambda: FILMS)

    def test_clicks_and_saves_count_as_relevance_but_opens_do_not(self) -> None:
        impressions = [("k1", MOOD, IDS), ("k2", MOOD, IDS)]
        interactions = [
            ("k1", "tt0000001", "open", 0),
            ("k1", "tt0000002", "imdb_click", 1),
            ("k2", "tt0000003", "save", 2),
            ("k2", "tt0000003", "open", 2),
        ]
        acted_on, usage = _export_module().implicit_signals(impressions, interactions)
        assert acted_on == {MOOD: {"film 2", "film 3"}}
        assert usage[MOOD]["film 1"] == {
            "shown": 2,
            "open": 1,
            "imdb_click": 0,
            "save": 0,
            "similar_click": 0,
            "mean_position": 0.0,
        }
        assert usage[MOOD]["film 3"]["save"] == 1 and usage[MOOD]["film 3"]["mean_position"] == 2.0

    def test_shown_and_ignored_films_are_counted(self) -> None:
        _, usage = _export_module().implicit_signals([("k1", MOOD, IDS)], [])
        assert {t: s["shown"] for t, s in usage[MOOD].items()} == {
            "film 1": 1,
            "film 2": 1,
            "film 3": 1,
        }
        assert "mean_position" not in usage[MOOD]["film 1"]

    def test_interactions_without_a_known_impression_or_film_are_dropped(self) -> None:
        acted_on, usage = _export_module().implicit_signals(
            [("k1", MOOD, IDS)],
            [("unknown-key", "tt0000001", "save", 0), ("k1", "tt9999999", "save", 0)],
        )
        assert acted_on == {}
        assert all(s["save"] == 0 for s in usage[MOOD].values())
