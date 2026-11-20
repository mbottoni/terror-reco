"""Only corpus strategies exist; retired inputs degrade instead of breaking.

The keyword strategy, the /loading interstitial and the `kind` filter were
removed together. Old bookmarks and shared links still carry those values, so
they must fall back to a normal search rather than 404 or 500.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as main_mod

FILM: dict[str, Any] = {
    "imdb_id": "tt0000001",
    "title": "Some Film",
    "overview": "A plot.",
    "poster_url": "https://example.com/p.jpg",
    "keywords": "isolation",
    "year": "1979",
}


@pytest.fixture()
def calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def _fake(**kwargs: Any) -> list[dict[str, Any]]:
        seen.append(kwargs)
        return [dict(FILM)]

    monkeypatch.setattr(main_mod, "recommend_movies_advanced", _fake)
    return seen


class TestRetiredInputs:
    def test_keyword_strategy_falls_back_to_a_corpus_search(
        self, client: TestClient, calls: list[dict[str, Any]]
    ) -> None:
        resp = client.get("/recommend", params={"mood": "tense", "strategy": "keyword"})
        assert resp.status_code == 200
        assert "Some Film" in resp.text
        assert "Keyword Match" not in resp.text
        assert len(calls) >= 1

    def test_kind_parameter_is_ignored_not_forwarded(
        self, client: TestClient, calls: list[dict[str, Any]]
    ) -> None:
        resp = client.get(
            "/recommend", params={"mood": "tense", "strategy": "semantic", "kind": "series"}
        )
        assert resp.status_code == 200
        assert all("kind" not in kwargs for kwargs in calls)

    def test_loading_page_is_gone(self, client: TestClient) -> None:
        assert client.get("/loading", params={"mood": "tense"}).status_code == 404


class TestSearchForm:
    def test_form_offers_only_live_options(self, client: TestClient) -> None:
        html = client.get("/").text
        assert 'action="/recommend"' in html
        assert 'value="keyword"' not in html
        assert 'name="kind"' not in html
        assert "/loading" not in html

    def test_unified_is_the_preselected_strategy(self, client: TestClient) -> None:
        html = client.get("/").text
        assert '<option value="unified" selected>' in html
        assert '<option value="semantic">' in html


class TestDefaultStrategy:
    """A request that names no strategy gets unified, the better one as served."""

    @pytest.fixture()
    def unified_calls(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        import app.main as main_mod

        seen: list[str] = []

        def _unified(**kwargs: Any) -> list[dict[str, Any]]:
            seen.append(kwargs["mood"])
            return []

        monkeypatch.setattr(main_mod, "recommend_movies_advanced", lambda **kw: [])
        monkeypatch.setattr(main_mod, "recommend_unified_semantic", _unified)
        return seen

    def test_no_strategy_means_unified(self, client: TestClient, unified_calls: list[str]) -> None:
        client.get("/recommend", params={"mood": "tense"})
        assert unified_calls == ["tense"]

    def test_an_unknown_strategy_means_unified(
        self, client: TestClient, unified_calls: list[str]
    ) -> None:
        client.get("/recommend", params={"mood": "tense", "strategy": "keyword"})
        assert unified_calls == ["tense"]

    def test_semantic_is_still_available_by_name(
        self, client: TestClient, unified_calls: list[str]
    ) -> None:
        client.get("/recommend", params={"mood": "tense", "strategy": "semantic"})
        assert unified_calls == []
