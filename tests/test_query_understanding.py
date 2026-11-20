"""Reading a query into what it wants, what it excludes, and when."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from app.services import unified_recommender as ur
from app.services.query_understanding import Intent, aspect_scores, understand

TAGS = [
    "slasher",
    "very gory",
    "little to no gore",
    "campy",
    "horror comedy",
    "genuinely terrifying",
    "not too scary",
    "jump scares",
    "atmospheric",
    "zombies",
    "slow burn",
    "fast-paced",
    "korean horror",
    "outer space",
    "bleak",
    "melancholy",
    "ghosts",
]


def read(query: str) -> Intent:
    return understand(query, TAGS)


class TestWants:
    def test_a_tag_named_outright(self) -> None:
        assert read("campy slasher").want == {"campy": 1.0, "slasher": 1.0}

    def test_plain_words_for_a_tag(self) -> None:
        assert read("campy fun with lots of blood").want == {"campy": 1.0, "very gory": 1.0}
        assert "zombies" in read("a zombie movie").want

    def test_hyphens_and_tag_names_of_several_words(self) -> None:
        assert "slow burn" in read("slow-burn dread").want
        assert "outer space" in read("trapped in outer space").want

    def test_nothing_recognised_is_an_empty_intent(self) -> None:
        intent = read("something to watch tonight")
        assert not intent and intent.want == {} and intent.years is None

    def test_a_loose_phrase_counts_by_its_closeness(self) -> None:
        def matcher(phrases: list[str]) -> dict[str, tuple[str, float]]:
            return {p: ("ghosts", 0.8 if p == "spectral" else 0.3) for p in phrases}

        assert understand("spectral visitors", TAGS, matcher).want == {"ghosts": 0.8}

    def test_a_word_matched_exactly_is_not_also_matched_loosely(self) -> None:
        asked: list[str] = []

        def matcher(phrases: list[str]) -> dict[str, tuple[str, float]]:
            asked.extend(phrases)
            return {p: ("slasher", 0.9) for p in phrases}

        assert understand("very gory", TAGS, matcher).want == {"very gory": 1.0}
        assert asked == []


class TestExclusions:
    def test_no_gore_avoids_gore_and_wants_its_opposite(self) -> None:
        intent = read("scary but with no gore")
        assert intent.avoid == {"very gory"}
        assert intent.want == {"genuinely terrifying": 1.0, "little to no gore": 0.8}

    def test_not_a_comedy(self) -> None:
        intent = read("zombies but not a comedy")
        assert intent.want == {"zombies": 1.0} and intent.avoid == {"horror comedy"}

    def test_more_than_names_the_thing_wanted_less(self) -> None:
        intent = read("funny more than scary")
        assert "horror comedy" in intent.want and intent.avoid == {"genuinely terrifying"}

    def test_a_negation_stops_at_but_and_at_a_comma(self) -> None:
        assert read("not gory but campy").want.get("campy") == 1.0
        intent = read("no jump scares, just atmosphere")
        assert intent.avoid == {"jump scares"} and "atmospheric" in intent.want

    def test_a_tag_whose_name_starts_with_not_is_wanted_not_excluded(self) -> None:
        intent = read("a ghost story that is not too scary")
        assert "not too scary" in intent.want and "not too scary" not in intent.avoid


class TestDecades:
    @pytest.mark.parametrize(
        ("query", "years"),
        [
            ("80s slasher", (1980, 1989)),
            ("horror from the 1930s", (1930, 1939)),
            ("something from the 2020s", (2020, 2029)),
            ("the eighties", (1980, 1989)),
            ("the 20s", (2020, 2029)),
            ("slasher", None),
        ],
    )
    def test_decades(self, query: str, years: tuple[int, int] | None) -> None:
        assert read(query).years == years

    def test_the_decade_is_not_also_read_as_a_quality(self) -> None:
        assert read("80s slasher").want == {"slasher": 1.0}


def test_place_adjectives_add_the_country_for_keyword_matching() -> None:
    assert read("korean horror").extra_terms == ("korea",)
    assert read("slasher").extra_terms == ()


class TestPeople:
    PEOPLE = frozenset({"john carpenter", "vincent price", "guillermo del toro"})

    def test_a_full_name_from_the_credits_is_recognised(self) -> None:
        intent = understand("horror with Vincent Price", TAGS, known_people=self.PEOPLE)
        assert intent.people == ("vincent price",) and intent
        assert understand("Guillermo del Toro ghosts", TAGS, known_people=self.PEOPLE).people == (
            "guillermo del toro",
        )

    def test_a_first_name_alone_is_not_a_person(self) -> None:
        assert (
            understand("a film about a man called john", TAGS, known_people=self.PEOPLE).people
            == ()
        )

    def test_films_without_the_person_are_marked(self) -> None:
        films = [
            {"director": "John Carpenter", "actors": "Kurt Russell"},
            {"director": "Roger Corman", "actors": "Vincent Price, Hazel Court"},
            {"director": None},
        ]
        coverage, violates = aspect_scores(Intent(people=("vincent price",)), films)
        assert coverage == pytest.approx([0.0, 1.0, 0.0]) and violates == [True, False, True]


class TestAspectScores:
    FILMS = [
        {"tags": "campy, very gory", "year": 1985},
        {"tags": "campy", "year": 1985},
        {"tags": "very gory, bleak", "year": 2015},
        {"tags": "", "year": None},
    ]

    def test_coverage_is_the_share_of_what_was_asked_for(self) -> None:
        coverage, violates = aspect_scores(
            Intent(want={"campy": 1.0, "very gory": 1.0}), self.FILMS
        )
        assert coverage == pytest.approx([1.0, 0.5, 0.5, 0.0])
        assert violates == [False] * 4

    def test_an_avoided_tag_or_the_wrong_decade_is_a_violation(self) -> None:
        intent = Intent(want={"very gory": 1.0}, avoid=frozenset({"bleak"}), years=(1980, 1989))
        coverage, violates = aspect_scores(intent, self.FILMS)
        assert coverage == pytest.approx([1.0, 0.5, 0.5, 0.0])
        assert violates == [False, False, True, True]


class TestInTheRanker:
    """The ranker's side: caching, the penalty, and never failing a request."""

    @pytest.fixture(autouse=True)
    def small_vocabulary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ur, "tag_definition_texts", lambda: (TAGS, [t + ": x" for t in TAGS]))
        monkeypatch.setattr(
            ur, "_embed_sbert", lambda texts: np.tile([1.0, 0, 0, 0], (len(texts), 1)).astype("f")
        )
        monkeypatch.setattr(ur, "_TAG_VECTOR_CACHE", {})
        ur._INTENT_CACHE.clear()

    def test_intent_is_cached_and_survives_a_broken_embedder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = ur.query_intent("80s slasher")
        assert first.years == (1980, 1989) and ur.query_intent("80s slasher") is first

        def broken(texts: list[str]) -> np.ndarray:
            raise RuntimeError("no model")

        monkeypatch.setattr(ur, "_embed_sbert", broken)
        assert not ur.query_intent("a query whose phrases need the model")

    def test_a_film_from_the_wrong_decade_ranks_below_one_from_the_right_one(self) -> None:
        def film(imdb_id: str, year: int) -> dict[str, Any]:
            return {
                "imdb_id": imdb_id,
                "title": imdb_id,
                "overview": "x",
                "tags": "slasher",
                "year": year,
            }

        items = [film("tt-wrong", 2015), film("tt-right", 1984)]
        ranked = ur.recommend_unified_semantic(
            mood="80s slasher", items=items, limit=2, temperature=0.0
        )
        assert [m["imdb_id"] for m in ranked] == ["tt-right", "tt-wrong"]

    def test_switched_off_it_reads_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ur, "QUERY_UNDERSTANDING", False)
        assert not ur.query_intent("80s slasher")
