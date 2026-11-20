"""The tag signal: how well a film's descriptive tags answer the query."""

from __future__ import annotations

import numpy as np
import pytest

from app.services import unified_recommender as ur
from app.services.corpus import load_tag_vocabulary


@pytest.fixture()
def two_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    """A vocabulary of two tags whose definitions embed to the two axes."""
    monkeypatch.setattr(
        ur, "tag_definition_texts", lambda: (["slasher", "folk horror"], ["a", "b"])
    )
    monkeypatch.setattr(ur, "_embed_sbert", lambda texts: np.eye(2, dtype=np.float32)[: len(texts)])
    monkeypatch.setattr(ur, "_TAG_VECTOR_CACHE", {})


def test_a_film_scores_by_its_best_matching_tag(two_tags: None) -> None:
    items = [
        {"tags": "slasher"},
        {"tags": "folk horror"},
        {"tags": "slasher, folk horror"},
        {"tags": ""},
        {"tags": "not in the vocabulary"},
    ]
    scores = ur.tag_scores(items, query_vec=np.array([[0.2, 0.9]], dtype=np.float32))
    assert scores.tolist() == pytest.approx([0.2, 0.9, 0.9, 0.0, 0.0])


def test_named_films_are_matched_on_shared_tags(two_tags: None) -> None:
    items = [{"tags": "slasher, gory"}, {"tags": "slasher"}, {"tags": "zombies"}, {"tags": ""}]
    scores = ur.tag_scores(items, like_tags={"slasher", "gory"})
    assert scores.tolist() == pytest.approx([1.0, 0.5, 0.0, 0.0])


def test_the_signal_drops_out_instead_of_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(texts: list[str]) -> np.ndarray:
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(ur, "_embed_sbert", broken)
    monkeypatch.setattr(ur, "_TAG_VECTOR_CACHE", {})
    scores = ur.tag_scores([{"tags": "slasher"}], query_vec=np.ones((1, 4), dtype=np.float32))
    assert scores.tolist() == [0.0]


def test_tag_vectors_are_rebuilt_when_the_embedder_changes(
    two_tags: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = ur._tag_vectors()
    assert first is not None
    monkeypatch.setattr(
        ur, "_embed_sbert", lambda texts: np.zeros((len(texts), 2), dtype=np.float32)
    )
    second = ur._tag_vectors()
    assert second is not None and float(second[1].sum()) == 0.0


def test_every_tag_has_a_meaning_to_embed() -> None:
    vocabulary = load_tag_vocabulary()
    assert len(vocabulary) >= 100
    assert all(meaning.strip() for meaning in vocabulary.values())


def test_tags_carry_weight_in_the_default_blend() -> None:
    assert ur.DEFAULT_WEIGHTS["tags"] > 0
    assert sum(ur.DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)
