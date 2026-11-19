"""Startup runs through the lifespan handler and never blocks boot."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as main_mod
from app.main import app


def test_startup_uses_lifespan_not_on_event() -> None:
    assert app.router.on_startup == []
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200


def test_boot_survives_a_missing_model_and_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.services.corpus as corpus_mod
    import app.services.unified_recommender as unified_mod

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("unavailable")

    monkeypatch.setattr(unified_mod, "_get_sbert", _boom)
    monkeypatch.setattr(corpus_mod, "load_corpus", _boom)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200


def test_corpus_is_not_encoded_at_startup_without_a_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.corpus as corpus_mod

    called: list[bool] = []
    monkeypatch.setattr(corpus_mod, "load_corpus", lambda: [{"imdb_id": "tt1"}])
    monkeypatch.setattr(corpus_mod, "load_cached_embeddings", lambda corpus: None)
    monkeypatch.setattr(
        corpus_mod, "get_corpus_and_embeddings", lambda: called.append(True) or ([], None)
    )
    main_mod._warm_model_and_corpus()
    assert called == []
