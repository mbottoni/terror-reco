"""Tests for the watchlist."""

from __future__ import annotations

import re
from typing import Any

from app.models import WatchlistItem


def _register(client: Any, email: str = "wl@example.com") -> None:
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


class TestWatchlistToggle:
    def test_requires_sign_in(self, client: Any) -> None:
        resp = client.post("/watchlist/toggle", json={"imdb_id": "tt1", "title": "X"})
        assert resp.status_code == 401

    def test_add_then_remove_is_idempotent_toggling(self, client: Any, db: Any) -> None:
        _register(client)
        added = client.post("/watchlist/toggle", json={"imdb_id": "tt1", "title": "X"}).json()
        assert added["saved"] is True
        assert db.query(WatchlistItem).count() == 1

        removed = client.post("/watchlist/toggle", json={"imdb_id": "tt1", "title": "X"}).json()
        assert removed["saved"] is False
        assert db.query(WatchlistItem).count() == 0

    def test_rejects_missing_imdb_id(self, client: Any) -> None:
        _register(client, "wl2@example.com")
        assert client.post("/watchlist/toggle", json={"title": "no id"}).status_code == 400

    def test_saving_twice_does_not_duplicate(self, client: Any, db: Any) -> None:
        """The unique constraint is the backstop; toggling is the behaviour."""
        _register(client, "wl3@example.com")
        client.post("/watchlist/toggle", json={"imdb_id": "tt7", "title": "A"})
        client.post("/watchlist/toggle", json={"imdb_id": "tt7", "title": "A"})
        client.post("/watchlist/toggle", json={"imdb_id": "tt7", "title": "A"})
        assert db.query(WatchlistItem).count() == 1


class TestWatchlistPage:
    def test_anonymous_is_redirected_to_login(self, client: Any) -> None:
        resp = client.get("/watchlist/", follow_redirects=False)
        assert resp.status_code == 303
        assert "/auth/login" in resp.headers["location"]

    def test_shows_saved_films(self, client: Any) -> None:
        _register(client, "wl4@example.com")
        client.post("/watchlist/toggle", json={"imdb_id": "tt42", "title": "Saved Film"})
        assert "Saved Film" in client.get("/watchlist/").text

    def test_empty_state(self, client: Any) -> None:
        _register(client, "wl5@example.com")
        assert "Nothing saved yet" in client.get("/watchlist/").text
