"""Growing the corpus: what is let in, what the free API budget allows, and
how new films get tags. Nothing here touches the network or the model."""

from __future__ import annotations

import asyncio
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import grow_corpus as grow
from scripts import tag_predictor as predictor


def _record(**changes: Any) -> dict[str, Any]:
    base = {
        "imdb_id": "tt1",
        "title": "A Film",
        "overview": "Something happens.",
        "poster_url": "https://image.tmdb.org/t/p/w500/x.jpg",
        "rating_source": "imdb",
        "imdbVotes": "12,345",
        "genre": "Horror, Thriller",
    }
    return {**base, **changes}


class TestWhatIsLetIn:
    def test_a_complete_film_is_accepted(self) -> None:
        assert grow.rejection_reason(_record(), min_imdb_votes=5000) is None

    @pytest.mark.parametrize(
        ("changes", "why"),
        [
            ({"overview": "  "}, "no overview"),
            ({"poster_url": None}, "no poster"),
            ({"rating_source": "tmdb"}, "no IMDb rating"),
            ({"imdbVotes": "4,999"}, "under 5000 IMDb votes"),
            ({"genre": "Comedy, Thriller"}, "not listed as horror"),
            ({"genre": "Animation, Family, Horror"}, "listed as family"),
            ({"genre": "Documentary, Horror"}, "listed as documentary"),
            ({"runtime": "14 min"}, "shorter than 60 minutes"),
        ],
    )
    def test_reasons(self, changes: dict[str, Any], why: str) -> None:
        assert grow.rejection_reason(_record(**changes), min_imdb_votes=5000) == why

    def test_a_rejection_expires(self) -> None:
        today = date(2026, 11, 20)
        recent = {"on": (today - timedelta(days=10)).isoformat()}
        old = {"on": (today - timedelta(days=grow.RETRY_REJECTED_AFTER_DAYS + 1)).isoformat()}
        assert grow.still_rejected(recent, today)
        assert not grow.still_rejected(old, today)
        assert not grow.still_rejected({"on": "not a date"}, today)


class _FakeTmdb:
    """Three pages of five films, ids 1..15, most voted first."""

    def __init__(self) -> None:
        self.pages_asked: list[int] = []

    async def discover_movies(
        self, *, page: int, min_votes: int, sort_by: str = "vote_count.desc"
    ) -> list[dict[str, Any]]:
        self.pages_asked.append(page)
        if page > 3:
            return []
        ids = range((page - 1) * 5 + 1, page * 5 + 1)
        # The best-rated list is a different set of films: 101, 102, ...
        return [{"id": i + (100 if "average" in sort_by else 0)} for i in ids]


class TestDiscovery:
    def test_skips_known_and_rejected_and_stops_when_it_has_enough(self) -> None:
        tmdb = _FakeTmdb()
        found, pages = asyncio.run(
            grow.discover_candidates(
                tmdb,
                known={1, 2, 3},
                skip={4, 6},
                want=4,
                min_tmdb_votes=100,
                orders=("vote_count.desc",),
            )
        )
        assert found == [5, 7, 8, 9]
        assert pages == 2 and tmdb.pages_asked == [1, 2]

    def test_alternates_between_the_most_voted_and_the_best_rated(self) -> None:
        found, _ = asyncio.run(
            grow.discover_candidates(_FakeTmdb(), known={1}, skip={101}, want=6, min_tmdb_votes=100)
        )
        assert found == [2, 102, 3, 103, 4, 104]

    def test_runs_out_of_films_gracefully(self) -> None:
        found, _ = asyncio.run(
            grow.discover_candidates(
                _FakeTmdb(),
                known=set(),
                skip=set(),
                want=99,
                min_tmdb_votes=100,
                orders=("vote_count.desc",),
            )
        )
        assert found == list(range(1, 16))

    def test_a_film_with_no_runtime_is_not_rejected_for_it(self) -> None:
        assert grow.rejection_reason(_record(runtime=None), min_imdb_votes=5000) is None


class TestOmdbBudget:
    def test_the_default_budget_is_well_inside_the_free_tier(self) -> None:
        assert grow.DEFAULT_OMDB_BUDGET <= grow.OMDB_DAILY_LIMIT // 2

    def test_a_quota_answer_stops_the_run_instead_of_rejecting_the_film(self) -> None:
        class Limited:
            async def get_by_id(self, imdb_id: str, *, plot_full: bool = False) -> dict[str, Any]:
                return {"Response": "False", "Error": "Request limit reached!"}

        with pytest.raises(grow.OmdbLimitReached):
            asyncio.run(grow.enrich(Limited(), _record()))


class TestTagPrediction:
    TAGS = ["slasher", "folk horror", "zombies", "very gory"]

    def _predict(self, new: list[list[float]]) -> list[list[str]]:
        # Two clusters of reference films: slashers on one axis, folk horror on the other.
        reference = np.array([[1, 0]] * 12 + [[0, 1]] * 12, dtype=np.float32)
        reference_tags = np.zeros((24, 4), dtype=np.float32)
        reference_tags[:12, [0, 3]] = 1  # slasher, very gory
        reference_tags[12:, 1] = 1  # folk horror
        tag_vecs = np.array([[1, 0], [0, 1], [0.7, 0.7], [0.9, 0.1]], dtype=np.float32)
        return predictor.predict_tags(
            np.array(new, dtype=np.float32), reference, reference_tags, tag_vecs, self.TAGS
        )

    def test_a_film_takes_the_tags_its_neighbours_agree_on(self) -> None:
        slasher, folk = self._predict([[1, 0], [0, 1]])
        assert {"slasher", "very gory"} <= set(slasher) and "folk horror" not in slasher[:2]
        assert folk[0] == "folk horror"

    def test_every_film_gets_a_bounded_number_of_tags_from_the_vocabulary(self) -> None:
        for tags in self._predict([[1, 0], [0, 1], [0.7, 0.7]]):
            assert predictor.MIN_TAGS <= len(tags) <= predictor.MAX_TAGS
            assert len(tags) == len(set(tags)) and set(tags) <= set(self.TAGS)

    def test_no_reference_films_means_no_guess(self) -> None:
        empty = np.zeros((0, 2), dtype=np.float32)
        out = predictor.predict_tags(
            np.ones((1, 2), dtype=np.float32), empty, np.zeros((0, 4)), np.ones((4, 2)), self.TAGS
        )
        assert out == [[]]


class TestTagFile:
    def test_rewriting_the_committed_file_changes_nothing(self, tmp_path: Path) -> None:
        from app.services.corpus import TAGS_FILE

        if not TAGS_FILE.exists():
            pytest.skip("no tag file in this environment")
        copy = tmp_path / "tags.json"
        predictor.write_tag_file(copy, json.loads(TAGS_FILE.read_text()))
        assert copy.read_text() == TAGS_FILE.read_text()

    def test_predicted_films_are_listed(self, tmp_path: Path) -> None:
        data = {
            "about": "x",
            "vocabulary": {},
            "tags": {"tt2": ["a"], "tt1": ["b"]},
            "predicted": ["tt2"],
        }
        out = tmp_path / "tags.json"
        predictor.write_tag_file(out, data)
        assert json.loads(out.read_text()) == data
        assert '  "tt2": ["a"]' in out.read_text()  # one film per line


def test_predicted_tags_never_outnumber_the_ones_someone_wrote() -> None:
    """Predicted tags are a stopgap. If they become most of the catalogue, the
    ranker's strongest signal is mostly guesses: time for a proper tagging pass."""
    from app.services.corpus import TAGS_FILE

    if not TAGS_FILE.exists():
        pytest.skip("no tag file in this environment")
    data = json.loads(TAGS_FILE.read_text())
    predicted = set(data.get("predicted", []))
    assert predicted <= set(data["tags"])
    assert len(predicted) <= len(data["tags"]) / 2, (
        f"{len(predicted)} of {len(data['tags'])} films carry predicted tags; "
        "replace them with a proper pass (see docs/corpus-growth.md)"
    )
