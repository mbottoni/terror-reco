"""One-click verdicts on a whole result set."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as main_mod
from app.main import app, set_result_key
from app.models import SetFeedback, User

IDS = ["tt0000001", "tt0000002", "tt0000003"]
BODY: dict[str, Any] = {"mood": "folk horror", "strategy": "unified", "imdb_ids": IDS, "verdict": 1}


def _post(client: Any, **overrides: Any) -> Any:
    return client.post("/api/set-feedback", json={**BODY, **overrides})


class TestRecording:
    def test_signed_out_visitor_can_leave_a_verdict(self, client: Any, db: Any) -> None:
        resp = _post(client)
        assert resp.status_code == 200
        assert resp.json() == {"status": "created", "verdict": 1}
        row = db.query(SetFeedback).one()
        assert (row.mood, row.strategy, row.imdb_ids, row.verdict) == (
            "folk horror",
            "unified",
            IDS,
            1,
        )
        assert row.user_id is None and len(row.visitor_id) == 32

    def test_same_verdict_again_removes_it(self, client: Any, db: Any) -> None:
        _post(client)
        assert _post(client).json() == {"status": "removed", "verdict": 0}
        assert db.query(SetFeedback).count() == 0

    def test_other_verdict_switches_it(self, client: Any, db: Any) -> None:
        _post(client)
        assert _post(client, verdict=-1).json() == {"status": "updated", "verdict": -1}
        db.expire_all()
        assert [r.verdict for r in db.query(SetFeedback).all()] == [-1]

    def test_different_sets_and_moods_are_separate_rows(self, client: Any, db: Any) -> None:
        _post(client)
        _post(client, imdb_ids=list(reversed(IDS)))  # order is part of the set
        _post(client, mood="slasher")
        assert db.query(SetFeedback).count() == 3

    def test_two_visitors_each_get_their_own_verdict(self, client: Any, db: Any) -> None:
        _post(client)
        with TestClient(app) as other:
            token = re.search(r'name="csrf-token" content="([^"]+)"', other.get("/").text)
            assert token
            resp = other.post("/api/set-feedback", json=BODY, headers={"X-CSRF-Token": token[1]})
            assert resp.json()["status"] == "created"
        assert db.query(SetFeedback).count() == 2
        assert len({r.visitor_id for r in db.query(SetFeedback).all()}) == 2

    def test_signed_in_verdict_is_linked_to_the_account(self, client: Any, db: Any) -> None:
        page = client.get("/auth/register").text
        csrf = re.search(r'name="csrf" value="([^"]+)"', page).group(1)  # type: ignore[union-attr]
        data = {
            "email": "rater@example.com",
            "password": "password123",
            "confirm_password": "password123",
            "csrf": csrf,
        }
        client.post("/auth/register", data=data)
        _post(client)
        assert db.query(SetFeedback).one().user_id == db.query(User).one().id


class TestValidation:
    @pytest.mark.parametrize(
        "override",
        [
            {"verdict": 0},
            {"verdict": 5},
            {"imdb_ids": []},
            {"imdb_ids": ["tt1"] * 21},
            {"imdb_ids": ["x" * 30]},
            {"mood": "   "},
        ],
    )
    def test_bad_bodies_are_422(self, client: Any, db: Any, override: dict[str, Any]) -> None:
        assert _post(client, **override).status_code == 422
        assert db.query(SetFeedback).count() == 0

    def test_requires_the_csrf_token(self, client: Any) -> None:
        resp = client.post("/api/set-feedback", json=BODY, headers={"X-CSRF-Token": "wrong"})
        assert resp.status_code == 403

    def test_result_key_depends_on_mood_and_order(self) -> None:
        assert set_result_key("Folk Horror ", IDS) == set_result_key("folk horror", IDS)
        assert set_result_key("folk horror", IDS) != set_result_key("folk horror", IDS[::-1])
        assert set_result_key("folk horror", IDS) != set_result_key("slasher", IDS)


class TestResultsPage:
    def test_control_is_present_but_hidden_until_script_runs(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        film = {"imdb_id": "tt0000001", "title": "A", "overview": "p", "keywords": "k"}
        monkeypatch.setattr(main_mod, "recommend_movies_advanced", lambda **kw: [film])
        page = client.get("/recommend", params={"mood": "x"}).text
        assert 'id="set-feedback" hidden' in page
        assert "function sendSetFeedback" in page and "/api/set-feedback" in page

    def test_no_control_on_an_empty_result(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(main_mod, "recommend_movies_advanced", lambda **kw: [])
        assert 'id="set-feedback"' not in client.get("/recommend", params={"mood": "x"}).text


def _export_module() -> Any:
    path = Path(__file__).resolve().parent.parent / "scripts" / "export_eval_set.py"
    spec = importlib.util.spec_from_file_location("export_eval_set", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestEvalExport:
    def test_grades_explicit_likes_above_set_level_thumbs_up(self) -> None:
        cases = _export_module().build_cases(
            liked={"folk horror": {"the wicker man"}},
            disliked={"folk horror": {"saw"}},
            set_liked={"folk horror": {"the wicker man", "midsommar", "the ritual", "saw"}},
            set_disliked={"folk horror": {"scream", "midsommar"}},
            min_likes=3,
        )
        assert len(cases) == 1
        case = cases[0]
        assert case["gold"] == ["midsommar", "the ritual", "the wicker man"]
        assert case["grades"] == {"midsommar": 1, "the ritual": 1, "the wicker man": 2}
        assert case["negative"] == ["saw"]  # an explicit dislike beats a set thumbs-up
        assert case["negative_weak"] == ["scream"]  # midsommar is gold, so not negative

    def test_set_verdicts_alone_can_make_a_case(self) -> None:
        cases = _export_module().build_cases({}, {}, {"slasher": {"a", "b", "c"}}, {}, 3)
        assert [c["mood"] for c in cases] == ["slasher"]

    def test_too_little_evidence_makes_no_case(self) -> None:
        assert _export_module().build_cases({"m": {"a"}}, {}, {"m": {"b"}}, {}, 3) == []

    def test_unknown_ids_are_dropped_when_resolving_titles(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import app.services.corpus as corpus_mod

        corpus = [{"imdb_id": "tt0000001", "title": "The Wicker Man"}]
        monkeypatch.setattr(corpus_mod, "load_corpus", lambda: corpus)
        up, down = _export_module().set_verdict_titles(
            [("folk horror", ["tt0000001", "tt9999999"], 1), ("folk horror", ["tt9999999"], -1)]
        )
        assert up == {"folk horror": {"the wicker man"}}
        assert dict(down) == {"folk horror": set()}
