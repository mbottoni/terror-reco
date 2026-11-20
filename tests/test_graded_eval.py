"""The graded benchmark's arithmetic and its data files.

The scores themselves need the model and are produced by
``scripts/graded_eval.py score``; these tests cover what CI can check: the
metric definitions, and that the committed moods and judgments are well formed.
"""

from __future__ import annotations

import json
import math

import pytest

from app.services.corpus import load_corpus
from scripts.graded_eval import (
    JUDGMENTS_FILE,
    bootstrap_ci,
    gain,
    graded_ndcg,
    holes,
    load_judgments,
    load_moods,
    precision_at,
)

GRADES = {"a": 2, "b": 2, "c": 1, "d": 0, "e": 1}


def test_gain_makes_a_square_fit_worth_three_partial_ones() -> None:
    assert (gain(0), gain(1), gain(2)) == (0.0, 1.0, 3.0)


def test_the_ideal_ordering_scores_one() -> None:
    assert graded_ndcg(["a", "b", "c", "e", "d"], GRADES) == pytest.approx(1.0)


def test_a_worse_ordering_of_the_same_films_scores_less() -> None:
    best = graded_ndcg(["a", "b", "c"], GRADES, k=3)
    worse = graded_ndcg(["c", "b", "a"], GRADES, k=3)
    assert worse < best


def test_ndcg_matches_a_hand_computation() -> None:
    # Top 2 are a film graded 1 and an unjudged film; the ideal top 2 is 2, 2.
    expected = (1.0 / math.log2(2)) / (3.0 / math.log2(2) + 3.0 / math.log2(3))
    assert graded_ndcg(["c", "zz"], GRADES, k=2) == pytest.approx(expected)


def test_unjudged_films_gain_nothing_and_are_counted_as_holes() -> None:
    assert graded_ndcg(["x", "y"], GRADES) == 0.0
    assert holes(["x", "a", "y"], GRADES) == 2
    assert holes(["a", "d"], GRADES) == 0  # a judged 0 is not a hole


def test_a_mood_with_no_relevant_film_scores_zero_not_an_error() -> None:
    assert graded_ndcg(["a"], {"a": 0}) == 0.0
    assert graded_ndcg(["a"], {}) == 0.0


def test_precision_counts_grades_at_or_above_the_floor() -> None:
    ranked = ["a", "c", "d", "x", "b", "e"]
    assert precision_at(ranked, GRADES, floor=1) == pytest.approx(4 / 6)
    assert precision_at(ranked, GRADES, floor=2) == pytest.approx(2 / 6)


def test_bootstrap_interval_contains_the_mean_and_is_reproducible() -> None:
    values = [0.1, 0.4, 0.5, 0.9, 0.3, 0.6]
    lo, hi = bootstrap_ci(values)
    assert lo < sum(values) / len(values) < hi
    assert bootstrap_ci(values) == (lo, hi)


def test_moods_are_distinct_and_the_hand_ones_match_the_notebook() -> None:
    from scripts.eval_gate import gold_cases

    moods = load_moods()
    everything = moods["hand"] + moods["pooled"] + moods["null"]
    assert len(everything) == len(set(everything))
    assert moods["hand"] == [case["mood"] for case in gold_cases()]
    assert len(moods["pooled"]) >= 30


@pytest.mark.skipif(not JUDGMENTS_FILE.exists(), reason="no judgments committed")
def test_judgments_are_well_formed() -> None:
    raw = json.loads(JUDGMENTS_FILE.read_text())
    assert raw["judge"], "the file must say who judged"
    moods = load_moods()
    known_moods = set(moods["hand"] + moods["pooled"])
    corpus_ids = {m["imdb_id"] for m in load_corpus()}
    judgments = load_judgments()
    assert set(judgments) <= known_moods
    for mood, rows in raw["judgments"].items():
        ids = [row["imdb_id"] for row in rows]
        assert len(ids) == len(set(ids)), f"duplicate film under {mood!r}"
        assert set(ids) <= corpus_ids, f"film not in the corpus under {mood!r}"
        assert {row["grade"] for row in rows} <= {0, 1, 2}


@pytest.mark.skipif(not JUDGMENTS_FILE.exists(), reason="no judgments committed")
def test_every_scoreable_mood_was_judged() -> None:
    moods = load_moods()
    judgments = load_judgments()
    missing = [m for m in moods["hand"] + moods["pooled"] if not judgments.get(m)]
    assert not missing, f"moods with no judgments: {missing}"
