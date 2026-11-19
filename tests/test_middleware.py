"""Security headers on every response, and per-client rate limits."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as main_mod
from app.middleware import DEFAULT_RULES, RateLimiter, Rule


class TestSecurityHeaders:
    @pytest.mark.parametrize("path", ["/", "/healthz", "/no-such-page", "/auth/login"])
    def test_headers_present_on_every_kind_of_response(self, client: TestClient, path: str) -> None:
        headers = client.get(path).headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]

    def test_no_hsts_when_debug(self, client: TestClient) -> None:
        # conftest forces DEBUG=true; HSTS over plain HTTP would be wrong.
        assert "Strict-Transport-Security" not in client.get("/").headers


class TestRateLimiter:
    def _limiter(self, limit: int = 3) -> tuple[RateLimiter, list[float]]:
        now = [1000.0]
        rule = Rule("t", frozenset({"GET"}), ("/x",), limit=limit, window=60.0)
        return RateLimiter(rules=(rule,), clock=lambda: now[0]), now

    def test_allows_up_to_the_limit_then_refuses(self) -> None:
        limiter, _ = self._limiter(limit=3)
        assert [limiter.check("GET", "/x", "a") for _ in range(3)] == [None, None, None]
        wait = limiter.check("GET", "/x", "a")
        assert wait is not None and 0 < wait <= 60

    def test_window_slides(self) -> None:
        limiter, now = self._limiter(limit=1)
        assert limiter.check("GET", "/x", "a") is None
        assert limiter.check("GET", "/x", "a") is not None
        now[0] += 61
        assert limiter.check("GET", "/x", "a") is None

    def test_clients_are_counted_separately(self) -> None:
        limiter, _ = self._limiter(limit=1)
        assert limiter.check("GET", "/x", "a") is None
        assert limiter.check("GET", "/x", "b") is None

    def test_unmatched_paths_and_methods_are_never_limited(self) -> None:
        limiter, _ = self._limiter(limit=1)
        for _ in range(5):
            assert limiter.check("GET", "/other", "a") is None
            assert limiter.check("POST", "/x", "a") is None

    def test_refused_requests_do_not_extend_the_window(self) -> None:
        limiter, now = self._limiter(limit=1)
        limiter.check("GET", "/x", "a")
        now[0] += 30
        assert limiter.check("GET", "/x", "a") is not None  # refused, not recorded
        now[0] += 31
        assert limiter.check("GET", "/x", "a") is None


def _limit(name: str) -> int:
    return next(r.limit for r in DEFAULT_RULES if r.name == name)


class TestRateLimitResponses:
    def test_login_is_limited_and_says_so_in_html(self, client: TestClient) -> None:
        data = {"email": "x@example.com", "password": "wrong-password", "csrf": "bad"}
        for _ in range(_limit("auth")):
            assert client.post("/auth/login", data=data, follow_redirects=False).status_code == 303
        blocked = client.post("/auth/login", data=data, follow_redirects=False)
        assert blocked.status_code == 429
        assert int(blocked.headers["Retry-After"]) >= 1
        assert "Slow down" in blocked.text
        assert blocked.headers["X-Content-Type-Options"] == "nosniff"

    def test_search_api_is_limited_with_json(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        film: dict[str, Any] = {"imdb_id": "tt1", "title": "A"}
        monkeypatch.setattr(main_mod, "recommend_movies_advanced", lambda **kw: [film])
        for _ in range(_limit("search")):
            assert client.get("/api/recommendations", params={"mood": "x"}).status_code == 200
        blocked = client.get("/api/recommendations", params={"mood": "x"})
        assert blocked.status_code == 429
        assert "detail" in blocked.json()

    def test_ordinary_pages_are_not_limited(self, client: TestClient) -> None:
        for _ in range(_limit("search") + 20):
            assert client.get("/").status_code == 200
