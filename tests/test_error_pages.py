"""Error pages, and the promise that ranking never runs on the event loop.

Both cover regressions that are invisible from a passing happy path: a missing
corpus used to escape the HTML route as a bare 500 traceback while the JSON API
answered the same condition with a clean 503, and the corpus pipeline used to be
declared ``async`` despite doing nothing but CPU work.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.services import corpus as corpus_mod
from app.services.recommender import recommend_movies_advanced, similar_movies


class TestNotAsync:
    """The corpus pipeline must stay synchronous.

    Declaring these ``async`` without an ``await`` in them puts a
    sentence-transformer forward pass and a corpus-wide matmul directly on the
    event loop, so concurrent requests serialise behind each other and even
    ``/healthz`` waits. They are handed to ``run_in_threadpool`` by the routes
    instead; re-adding ``async`` here would silently undo that.
    """

    def test_recommend_movies_advanced_is_not_a_coroutine(self) -> None:
        assert not inspect.iscoroutinefunction(recommend_movies_advanced)

    def test_similar_movies_is_not_a_coroutine(self) -> None:
        assert not inspect.iscoroutinefunction(similar_movies)


@pytest.fixture()
def _no_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a deploy that shipped without a corpus."""
    monkeypatch.setattr(
        corpus_mod, "get_corpus_and_embeddings", lambda: ([], np.zeros((0, 0), dtype=np.float32))
    )


class TestMissingCorpus:
    def test_html_route_renders_a_page_not_a_traceback(
        self, client: TestClient, _no_corpus: Any
    ) -> None:
        resp = client.get("/recommend", params={"mood": "haunted house"})
        assert resp.status_code == 503
        assert "text/html" in resp.headers["content-type"]
        assert "corpus is not available" in resp.text

    def test_json_api_still_answers_in_json(self, client: TestClient, _no_corpus: Any) -> None:
        resp = client.get("/api/recommendations", params={"mood": "haunted house"})
        assert resp.status_code == 503
        assert resp.json()["detail"]

    def test_healthz_agrees_with_the_error_page(self, client: TestClient, _no_corpus: Any) -> None:
        """A probe reporting healthy while every search 503s is worse than useless."""
        assert client.get("/healthz").json()["status"] == "degraded"


class TestNotFound:
    def test_unknown_page_renders_html(self, client: TestClient) -> None:
        resp = client.get("/no-such-page")
        assert resp.status_code == 404
        assert "text/html" in resp.headers["content-type"]
        assert "Page not found" in resp.text

    def test_unknown_api_path_stays_json(self, client: TestClient) -> None:
        """An HTML body would break any client parsing the JSON API."""
        resp = client.get("/api/no-such-endpoint")
        assert resp.status_code == 404
        assert "application/json" in resp.headers["content-type"]

    def test_known_api_404_keeps_its_detail(self, client: TestClient, _no_corpus: Any) -> None:
        resp = client.get("/api/similar/tt0000000")
        assert resp.status_code == 404
        assert "application/json" in resp.headers["content-type"]
