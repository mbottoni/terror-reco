"""Where to watch: a dated snapshot, looked up without any network call."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.services.providers as providers
from app.services.providers import providers_for, region_from_accept_language

SNAPSHOT = {
    "fetched_at": "2026-11-20T01:47:32+00:00",
    "regions": ["US", "GB"],
    "films": {
        "tt0000001": {
            "tmdb": 123,
            "US": {"stream": ["Shudder"], "rent": ["Apple TV Store"], "buy": []},
            "GB": {"stream": [], "rent": [], "buy": ["Amazon Video"]},
        },
        "tt0000002": {"tmdb": 456, "US": {"stream": [], "rent": [], "buy": []}},
    },
}


@pytest.fixture(autouse=True)
def snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    file = tmp_path / "watch_providers.json"
    file.write_text(json.dumps(SNAPSHOT))
    monkeypatch.setattr(providers, "PROVIDERS_FILE", file)
    monkeypatch.setattr(providers, "_CACHE", {})
    return file


class TestLookup:
    def test_returns_the_region_entry_with_its_date_and_link(self) -> None:
        assert providers_for("tt0000001", "US") == {
            "region": "US",
            "as_of": "2026-11-20",
            "stream": ["Shudder"],
            "rent": ["Apple TV Store"],
            "buy": [],
            "link": "https://www.themoviedb.org/movie/123/watch?locale=US",
        }

    @pytest.mark.parametrize(
        ("imdb_id", "region"),
        [
            ("tt9999999", "US"),  # film not in the snapshot
            ("tt0000001", "DE"),  # no data for that region
            ("tt0000002", "US"),  # an entry with nothing in it
            ("tt0000001", "XX"),  # a region that is not stored at all
            ("tt0000001", "tmdb"),  # a key that is not a region
        ],
    )
    def test_nothing_known_is_none(self, imdb_id: str, region: str) -> None:
        assert providers_for(imdb_id, region) is None

    def test_missing_or_corrupt_file_means_no_data(self, snapshot: Path) -> None:
        snapshot.write_text("{not json")
        assert providers_for("tt0000001", "US") is None
        snapshot.unlink()
        providers._CACHE.clear()
        assert providers_for("tt0000001", "US") is None

    def test_file_is_reread_when_it_changes(self, snapshot: Path) -> None:
        import os

        assert providers_for("tt0000001", "US") is not None
        changed: dict[str, Any] = {**SNAPSHOT, "films": {}}
        snapshot.write_text(json.dumps(changed))
        os.utime(snapshot, (snapshot.stat().st_atime, snapshot.stat().st_mtime + 5))
        assert providers_for("tt0000001", "US") is None


class TestRegionGuess:
    @pytest.mark.parametrize(
        ("header", "expected"),
        [
            ("en-GB,en;q=0.9", "GB"),
            ("pt-BR,pt;q=0.9,en;q=0.8", "BR"),
            ("pt", "BR"),
            ("de", "DE"),
            ("fr-CH, fr;q=0.9", "FR"),  # Switzerland is not stored; French implies FR
            ("en-US,en;q=0.9", "US"),
            ("en", "US"),
            ("ja-JP", "US"),  # nothing stored for Japan
            ("", "US"),
            (None, "US"),
            ("en_AU", "AU"),
            ("garbage;;;,,", "US"),
        ],
    )
    def test_guess(self, header: str | None, expected: str) -> None:
        assert region_from_accept_language(header) == expected


class TestApi:
    def test_uses_the_browsers_language_for_the_region(self, client: TestClient) -> None:
        resp = client.get("/api/providers/tt0000001", headers={"Accept-Language": "en-GB"})
        assert resp.status_code == 200
        assert resp.json()["region"] == "GB" and resp.json()["buy"] == ["Amazon Video"]

    def test_explicit_region_wins(self, client: TestClient) -> None:
        resp = client.get(
            "/api/providers/tt0000001?region=us", headers={"Accept-Language": "en-GB"}
        )
        assert resp.json()["region"] == "US" and resp.json()["stream"] == ["Shudder"]

    def test_unknown_film_or_region_is_404(self, client: TestClient) -> None:
        assert client.get("/api/providers/tt9999999").status_code == 404
        assert client.get("/api/providers/tt0000001?region=DE").status_code == 404

    def test_malformed_region_is_422(self, client: TestClient) -> None:
        assert client.get("/api/providers/tt0000001?region=USA").status_code == 422


class TestFetchScriptShaping:
    def _module(self) -> Any:
        import importlib.util

        path = Path(__file__).resolve().parent.parent / "scripts" / "fetch_watch_providers.py"
        spec = importlib.util.spec_from_file_location("fetch_watch_providers", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_orders_by_priority_and_drops_reseller_duplicates(self) -> None:
        raw = {
            "flatrate": [
                {"provider_name": "Shudder Amazon Channel", "display_priority": 1},
                {"provider_name": "Netflix Standard with Ads", "display_priority": 2},
                {"provider_name": "Shudder", "display_priority": 5},
                {"provider_name": "Netflix", "display_priority": 3},
                {"provider_name": "Netflix", "display_priority": 4},
            ],
            "free": [{"provider_name": "Tubi TV", "display_priority": 9}],
            "rent": [],
        }
        assert self._module().shape_region(raw) == {
            "stream": ["Netflix", "Shudder", "Tubi TV"],
            "rent": [],
            "buy": [],
        }

    def test_region_with_nothing_is_dropped(self) -> None:
        assert self._module().shape_region({"link": "https://example.com"}) is None


class TestShippedSnapshot:
    def test_real_file_matches_the_corpus_and_the_expected_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.services.corpus import load_corpus

        real = Path(__file__).resolve().parent.parent / "data" / "watch_providers.json"
        if not real.exists() or not load_corpus():
            pytest.skip("snapshot or corpus not present in this environment")
        data = json.loads(real.read_text())
        corpus_ids = {m["imdb_id"] for m in load_corpus()}
        assert set(data["films"]) <= corpus_ids
        assert len(data["films"]) > len(corpus_ids) * 0.8
        assert data["fetched_at"][:2] == "20"
        for film in data["films"].values():
            assert isinstance(film["tmdb"], int)
            for region, entry in film.items():
                if region != "tmdb":
                    assert region in providers.REGIONS
                    assert set(entry) == {"stream", "rent", "buy"}
