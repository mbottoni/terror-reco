"""Ranking regression gate: replays a frozen slice of the evaluation.

See ``scripts/eval_gate.py`` for what the fixture holds and why. Nothing here
loads the model, so this runs in CI on every push.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from scripts.eval_gate import (
    FIXTURE_META,
    FIXTURE_VECS,
    evaluate,
    ndcg_at_k,
    stored_vectors,
    tag_lookup,
    text_sha,
    title_match,
)

REBUILD = (
    "If the change to the corpus or to embedding_text() is intended, run "
    "`python scripts/eval_gate.py` to rebuild the fixture, and `make eval` to see what "
    "the change did to ranking before committing both."
)


@pytest.fixture(scope="module")
def gate() -> dict[str, Any]:
    from app.services.corpus import load_corpus

    corpus = load_corpus()
    if not corpus:
        pytest.skip("corpus not built in this environment")
    meta = json.loads(FIXTURE_META.read_text())
    vecs = np.load(FIXTURE_VECS)
    by_id = {m["imdb_id"]: m for m in corpus}
    return {"meta": meta, "vecs": vecs, "by_id": by_id}


class TestFixtureStillDescribesTheCorpus:
    """The stored vectors are only valid for the exact text they were made from."""

    def test_every_fixture_film_is_still_in_the_corpus(self, gate: dict[str, Any]) -> None:
        missing = [f["imdb_id"] for f in gate["meta"]["films"] if f["imdb_id"] not in gate["by_id"]]
        assert not missing, f"{len(missing)} fixture films left the corpus. {REBUILD}"

    def test_embedded_text_is_unchanged(self, gate: dict[str, Any]) -> None:
        """Catches a corpus field going missing (the keyword-loss regression)
        and any change to how the embedded document is composed."""
        from app.services.corpus import embedding_text
        from app.services.unified_recommender import _normalize_text

        changed = [
            f["imdb_id"]
            for f in gate["meta"]["films"]
            if f["imdb_id"] in gate["by_id"]
            and text_sha(_normalize_text(embedding_text(gate["by_id"][f["imdb_id"]])))
            != f["text_sha"]
        ]
        total = len(gate["meta"]["films"])
        assert not changed, (
            f"The text that gets embedded changed for {len(changed)} of {total} fixture films, "
            f"so their stored vectors no longer match production. {REBUILD}"
        )

    def test_vectors_were_built_with_the_current_recipe(self, gate: dict[str, Any]) -> None:
        """The text can be unchanged while the way vectors are made from it is not."""
        from app.services.corpus import EMBEDDING_RECIPE

        assert gate["meta"].get("recipe") == EMBEDDING_RECIPE, (
            f"The fixture was built with recipe {gate['meta'].get('recipe')!r}, production "
            f"uses {EMBEDDING_RECIPE!r}. {REBUILD}"
        )


class TestRankingHasNotCollapsed:
    def test_ndcg_stays_near_the_recorded_baseline(self, gate: dict[str, Any]) -> None:
        import app.services.corpus as corpus_mod
        from app.services.unified_recommender import _normalize_text, tag_definition_texts

        meta, vecs = gate["meta"], gate["vecs"]
        films = [
            gate["by_id"][f["imdb_id"]] for f in meta["films"] if f["imdb_id"] in gate["by_id"]
        ]
        if len(films) != len(meta["films"]):
            pytest.skip("fixture films missing from the corpus; reported by the test above")
        film_vecs = vecs["film_vecs"].astype(np.float32)
        lookup = {
            text_sha(_normalize_text(case["mood"])): vec.astype(np.float32)
            for case, vec in zip(meta["moods"], vecs["mood_vecs"], strict=True)
        }

        _, tag_texts = tag_definition_texts()
        assert [text_sha(text) for text in tag_texts] == meta["tag_texts"], (
            "The tag vocabulary changed since the fixture was built. " + REBUILD
        )
        lookup.update(tag_lookup(tag_texts, vecs["tag_vecs"]))

        with stored_vectors(corpus_mod, films, film_vecs, lookup):
            scores = evaluate(meta["moods"], films, film_vecs)

        for strategy, baseline in meta["baseline"].items():
            floor = baseline - meta["tolerance"]
            assert scores[strategy] >= floor, (
                f"{strategy} NDCG@{meta['k']} fell to {scores[strategy]:.4f}; the recorded "
                f"baseline is {baseline:.4f} and the floor is {floor:.4f}. This is a ranking "
                "regression on the frozen eval slice, not noise: inputs are deterministic."
            )

    def test_stored_vectors_are_restored_afterwards(self) -> None:
        import app.services.corpus as corpus_mod
        import app.services.unified_recommender as unified_mod

        before = (unified_mod._embed_sbert, corpus_mod.get_corpus_and_embeddings)
        with stored_vectors(corpus_mod, [], np.zeros((0, 0), dtype=np.float32), {}):
            assert unified_mod._embed_sbert is not before[0]
        assert (unified_mod._embed_sbert, corpus_mod.get_corpus_and_embeddings) == before

    def test_unknown_text_raises_instead_of_scoring_zero(self) -> None:
        import app.services.corpus as corpus_mod
        import app.services.unified_recommender as unified_mod

        with stored_vectors(corpus_mod, [], np.zeros((0, 0), dtype=np.float32), {}):
            with pytest.raises(KeyError, match="no vector"):
                unified_mod._embed_sbert(["a mood the fixture never saw"])


class TestScoring:
    def test_ndcg_perfect_and_empty(self) -> None:
        gold = {"a", "b"}
        assert ndcg_at_k(["a", "b", "x"], gold, n_gold=2) == pytest.approx(1.0)
        assert ndcg_at_k(["x", "y", "z"], gold, n_gold=2) == 0.0

    def test_ndcg_rewards_earlier_hits(self) -> None:
        gold = {"a"}
        assert ndcg_at_k(["a", "x"], gold, n_gold=1) > ndcg_at_k(["x", "a"], gold, n_gold=1)

    def test_title_match_follows_the_notebook_rule(self) -> None:
        assert title_match("The Witch", ["the witch"])
        assert title_match("Witch", ["the witch"])
        assert not title_match("Hereditary", ["the witch"])
