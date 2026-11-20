"""Descriptive tags: the committed data file, and how the corpus uses it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import app.services.corpus as corpus_mod
from app.services.corpus import TAGS_FILE, keyword_text, load_corpus, load_film_tags
from app.services.recommender import explain_match


@pytest.fixture(scope="module")
def tag_file() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(TAGS_FILE.read_text())
    return data


def test_the_file_says_who_wrote_the_tags(tag_file: dict[str, Any]) -> None:
    assert "AI model" in tag_file["about"]


def test_every_tag_comes_from_the_vocabulary(tag_file: dict[str, Any]) -> None:
    allowed = {entry["tag"] for facet in tag_file["vocabulary"].values() for entry in facet}
    assert len(allowed) >= 80
    stray = {t for tags in tag_file["tags"].values() for t in tags} - allowed
    assert not stray, f"tags outside the vocabulary: {sorted(stray)}"


def test_no_film_repeats_a_tag_or_carries_too_many(tag_file: dict[str, Any]) -> None:
    for imdb_id, tags in tag_file["tags"].items():
        assert len(tags) == len(set(tags)), imdb_id
        assert 1 <= len(tags) <= 10, imdb_id


def test_nearly_every_corpus_film_is_tagged() -> None:
    """A corpus rebuild adds films with no tags; this says when to tag them."""
    corpus = load_corpus()
    if not corpus:
        pytest.skip("corpus not built in this environment")
    tagged = sum(1 for m in corpus if m.get("tags"))
    assert tagged >= len(corpus) * 0.95, f"only {tagged}/{len(corpus)} films have tags"


def test_load_corpus_attaches_tags_as_a_comma_separated_field(tag_file: dict[str, Any]) -> None:
    corpus = load_corpus()
    if not corpus:
        pytest.skip("corpus not built in this environment")
    film = next(m for m in corpus if m["imdb_id"] in tag_file["tags"])
    assert film["tags"] == ", ".join(tag_file["tags"][film["imdb_id"]])


def test_a_missing_or_broken_tags_file_is_not_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(corpus_mod, "TAGS_FILE", tmp_path / "absent.json")
    assert load_film_tags() == {}
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    monkeypatch.setattr(corpus_mod, "TAGS_FILE", broken)
    assert load_film_tags() == {}


def test_saving_the_corpus_does_not_write_tags_into_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(corpus_mod, "CORPUS_DIR", tmp_path)
    monkeypatch.setattr(corpus_mod, "CORPUS_FILE", tmp_path / "corpus.json")
    monkeypatch.setattr(corpus_mod, "EMBEDDINGS_FILE", tmp_path / "e.npy")
    monkeypatch.setattr(corpus_mod, "EMBEDDINGS_META_FILE", tmp_path / "e.meta.json")
    corpus_mod.save_corpus([{"imdb_id": "tt1", "title": "A", "overview": "x", "tags": "campy"}])
    assert json.loads((tmp_path / "corpus.json").read_text()) == [
        {"imdb_id": "tt1", "title": "A", "overview": "x"}
    ]


def test_keyword_text_puts_tags_before_keywords() -> None:
    assert keyword_text({"tags": "slow burn, folk horror", "keywords": "island"}) == (
        "slow burn, folk horror, island"
    )
    assert keyword_text({"keywords": "island"}) == "island"
    assert keyword_text({"tags": "campy"}) == "campy"
    assert keyword_text({}) == ""


def test_tags_change_the_embedding_cache_key() -> None:
    film = {"imdb_id": "tt1", "title": "A", "overview": "x", "keywords": "k"}
    assert corpus_mod.corpus_fingerprint([film]) != corpus_mod.corpus_fingerprint(
        [{**film, "tags": "campy"}]
    )


def test_tags_are_not_in_the_long_embedded_document() -> None:
    """Measured: adding them there loses most of the gain."""
    film = {"imdb_id": "tt1", "title": "A", "overview": "x", "tags": "campy"}
    assert "campy" not in corpus_mod.embedding_text(film)


def test_matched_tags_are_shown_as_reasons_before_keywords() -> None:
    film = {"tags": "slow burn, folk horror", "keywords": "island, horror host"}
    assert explain_match("slow burn folk horror", film) == [
        "slow burn",
        "folk horror",
        "horror host",
    ]
