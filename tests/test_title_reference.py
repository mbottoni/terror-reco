"""Queries that name a film are searched with that film's vector."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.services import title_reference as tr
from app.services.corpus import load_corpus, semantic_search
from app.services.title_reference import find_title_references, normalise, reference_vector

ROOT = Path(__file__).resolve().parent.parent


def _film(
    imdb_id: str, title: str, year: int, votes: str = "1,000", **extra: Any
) -> dict[str, Any]:
    return {"imdb_id": imdb_id, "title": title, "year": year, "imdbVotes": votes, **extra}


CORPUS = [
    _film("tt1", "Hereditary", 2018),
    _film("tt2", "The Witch", 2016),
    _film("tt3", "Halloween", 1978, "300,000", collection_id=9),
    _film("tt4", "Halloween", 2018, "150,000", collection_id=9),
    _film("tt5", "Halloween II", 1981, collection_id=9),
    _film("tt6", "It", 2017),
    _film("tt7", "A Quiet Place", 2018),
    _film("tt8", "The Texas Chain Saw Massacre", 1974),
    _film("tt9", "Alien", 1979),
    _film("tt10", "The Thing", 1982),
    _film("tt11", "Smile", 2022),
    _film("tt12", "Mother!", 2017),
]


def _ids(query: str) -> tuple[str, ...] | None:
    ref = find_title_references(query, CORPUS)
    return ref.imdb_ids if ref else None


def test_normalise() -> None:
    assert normalise("Mother!") == "mother"
    assert normalise("Rosemary's  Baby") == "rosemarys baby"
    assert normalise("Häxan") == "haxan"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Hereditary", ("tt1",)),
        ("hereditary movie", ("tt1",)),
        ("the witch", ("tt2",)),
        ("movies like Hereditary", ("tt1",)),
        ("something similar to the witch but funnier", ("tt2",)),
        ("in the vein of a quiet place", ("tt7",)),
        ("texas chain saw massacre", ("tt8",)),  # leading "the" optional on long titles
        ("alien but underwater", ("tt9",)),
        ("mother!", ("tt12",)),
        ("hereditery", ("tt1",)),  # a misspelling, where a title is expected
    ],
)
def test_titles_are_found(query: str, expected: tuple[str, ...]) -> None:
    assert _ids(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        "a quiet place to hide from monsters",  # a title inside a sentence, no cue
        "witch in the woods",
        "halloween",  # alone, a common word is a mood
        "smile",
        "it",
        "I like it gory",  # "like" + a function word, not capitalised
        "feels like a nightmare",
        "the thing in the basement",
        "slow-burn psychological dread",
        "",
    ],
)
def test_ordinary_moods_are_left_alone(query: str) -> None:
    assert _ids(query) is None


def test_common_words_count_after_a_cue_or_with_a_year() -> None:
    assert _ids("like smile") == ("tt11",)
    assert _ids("movies like It") == ("tt6",)
    assert _ids("halloween 1978") == ("tt3",)


def test_same_title_picks_the_stated_year_else_the_best_known() -> None:
    assert _ids("like halloween") == ("tt3",)
    assert _ids("like halloween 2018") == ("tt4",)


def test_two_films() -> None:
    assert set(_ids("alien meets the thing") or ()) == {"tt9", "tt10"}


def test_the_rest_of_the_franchise_is_excluded_too() -> None:
    ref = find_title_references("like halloween", CORPUS)
    assert ref is not None
    assert ref.exclude_ids == {"tt3", "tt4", "tt5"}
    assert ref.titles == ("Halloween (1978)",)


def test_remakes_and_subtitled_entries_outside_the_collection_are_excluded() -> None:
    corpus = CORPUS + [
        _film("tt20", "Alien: Covenant", 2017, collection_id=77),  # filed as its own franchise
        _film("tt21", "Alien 3", 1992),
        _film("tt22", "It Follows", 2015),
        _film("tt23", "It Chapter Two", 2019),
    ]
    alien = find_title_references("like alien", corpus)
    assert alien is not None
    assert {"tt20", "tt21"} <= alien.exclude_ids
    it = find_title_references("movies like It", corpus)
    assert it is not None
    assert "tt23" in it.exclude_ids  # a numbered entry
    assert "tt22" not in it.exclude_ids  # a different film that starts with the same word


def test_residual_drops_the_title_cue_and_filler() -> None:
    ref = find_title_references("something like Hereditary but less bleak", CORPUS)
    assert ref is not None
    assert ref.residual == "less bleak"


def test_reference_vector_is_the_film_blended_with_the_rest() -> None:
    emb = np.eye(len(CORPUS), dtype=np.float32)
    rest = np.zeros((1, len(CORPUS)), dtype=np.float32)
    rest[0, 5] = 1.0

    alone = find_title_references("Hereditary", CORPUS)
    assert alone is not None
    vec = reference_vector(alone, emb, lambda _text: rest)
    assert vec.shape == (1, len(CORPUS))
    assert vec[0, 0] == pytest.approx(1.0)  # nothing else said: the film's own vector

    blended = find_title_references("like Hereditary but less bleak", CORPUS)
    assert blended is not None
    vec = reference_vector(blended, emb, lambda _text: rest)
    assert np.linalg.norm(vec) == pytest.approx(1.0)
    assert vec[0, 0] > vec[0, 5] > 0  # mostly the film, partly the rest


def test_search_leaves_the_named_film_and_its_franchise_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emb = np.eye(len(CORPUS), dtype=np.float32)
    emb[4] = emb[2]  # the sequel is the closest thing to the named film
    for temperature in (0.0, 1.0):
        found = semantic_search(
            "like halloween", CORPUS, emb, top_k=60, temperature=temperature, seed=1
        )
        ids = {m["imdb_id"] for m in found}
        assert not ids & {"tt3", "tt4", "tt5"}
        assert len(found) == len(CORPUS) - 3


def test_switch_turns_it_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tr, "ENABLED", False)
    assert _ids("Hereditary") is None


# --- against the real corpus ---------------------------------------------------
needs_corpus = pytest.mark.skipif(not load_corpus(), reason="corpus not built in this environment")


@needs_corpus
def test_no_benchmark_mood_or_home_page_chip_is_read_as_a_title() -> None:
    """A false reference replaces a mood search with "films like X". The mood
    benchmark is the guard: nothing in it may trigger."""
    corpus = load_corpus()
    moods = json.loads((ROOT / "eval" / "moods.json").read_text())
    queries = moods["hand"] + moods["pooled"] + moods["null"]
    queries += re.findall(r"pickMood\('([^']+)'\)", (ROOT / "app/templates/index.html").read_text())
    found = {q: find_title_references(q, corpus) for q in queries}
    hijacked = {q: ref.titles for q, ref in found.items() if ref is not None}
    assert not hijacked


@needs_corpus
def test_every_title_benchmark_query_is_detected() -> None:
    corpus = load_corpus()
    titles = json.loads((ROOT / "eval" / "moods.json").read_text()).get("title", [])
    assert titles, "the title-reference cases are missing from eval/moods.json"
    missed = [q for q in titles if find_title_references(q, corpus) is None]
    assert not missed
