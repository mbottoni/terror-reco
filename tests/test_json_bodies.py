"""Malformed JSON bodies get a 4xx that names the problem, never a 500."""

from __future__ import annotations

import re
from typing import Any

import pytest

from app.models import MovieFeedback


def _register(client: Any, email: str) -> None:
    page = client.get("/auth/register").text
    csrf = re.search(r'name="csrf" value="([^"]+)"', page).group(1)  # type: ignore[union-attr]
    client.post(
        "/auth/register",
        data={
            "email": email,
            "password": "password123",
            "confirm_password": "password123",
            "csrf": csrf,
        },
        follow_redirects=True,
    )


ENDPOINTS = ["/api/feedback", "/watchlist/toggle"]


@pytest.mark.parametrize("url", ENDPOINTS)
class TestMalformedBodies:
    @pytest.mark.parametrize("body", ["[]", '"x"', "3", "null"])
    def test_non_object_json_is_422(self, client: Any, url: str, body: str) -> None:
        _register(client, "shape@example.com")
        resp = client.post(url, content=body, headers={"Content-Type": "application/json"})
        assert resp.status_code == 422

    def test_invalid_json_is_422(self, client: Any, url: str) -> None:
        _register(client, "broken@example.com")
        resp = client.post(url, content="{not json", headers={"Content-Type": "application/json"})
        assert resp.status_code == 422

    def test_overlong_imdb_id_is_422(self, client: Any, url: str) -> None:
        _register(client, "long@example.com")
        resp = client.post(url, json={"imdb_id": "t" * 21, "rating": 1})
        assert resp.status_code == 422


class TestFeedback:
    @pytest.mark.parametrize("rating", [0, 2, "up", None])
    def test_rating_must_be_plus_or_minus_one(self, client: Any, rating: Any) -> None:
        _register(client, "rating@example.com")
        resp = client.post("/api/feedback", json={"imdb_id": "tt1", "rating": rating})
        assert resp.status_code == 422

    def test_valid_feedback_round_trip(self, client: Any, db: Any) -> None:
        _register(client, "ok@example.com")
        body = {"imdb_id": " tt1 ", "title": "X", "rating": 1, "mood": "m" * 600}
        created = client.post("/api/feedback", json=body)
        assert created.status_code == 200
        assert created.json() == {"status": "created", "imdb_id": "tt1", "rating": 1}
        row = db.query(MovieFeedback).one()
        assert row.imdb_id == "tt1"
        assert len(row.mood) == 512  # clipped to the column, not rejected

        switched = client.post("/api/feedback", json={"imdb_id": "tt1", "rating": -1}).json()
        assert switched["status"] == "updated"
        removed = client.post("/api/feedback", json={"imdb_id": "tt1", "rating": -1}).json()
        assert removed == {"status": "removed", "imdb_id": "tt1", "rating": 0}

    def test_signed_out_is_401(self, client: Any) -> None:
        resp = client.post("/api/feedback", json={"imdb_id": "tt1", "rating": 1})
        assert resp.status_code == 401
