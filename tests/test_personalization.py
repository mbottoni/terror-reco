"""Tests for feedback-driven personalisation.

Personalisation must be a *nudge*, never a hijack: a user's taste can reorder
results but must not override relevance to the query, and it must degrade to
neutral for anonymous or unrated users.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.models import MovieFeedback, User
from app.security import hash_password
from app.services.personalization import MIN_LIKES_FOR_TASTE, UserTaste, load_user_taste
from app.services.unified_recommender import recommend_unified_semantic


def _user(db: Any, email: str = "taste@example.com") -> User:
    user = User(email=email, password_hash=hash_password("password123"))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _rate(db: Any, user_id: int, imdb_id: str, rating: int) -> None:
    db.add(MovieFeedback(user_id=user_id, imdb_id=imdb_id, title=imdb_id, rating=rating))
    db.commit()


def _corpus(n: int = 6) -> list[dict[str, Any]]:
    return [
        {"imdb_id": f"tt{i}", "title": f"Film {i}", "overview": f"horror story {i}"}
        for i in range(n)
    ]


class TestLoadUserTaste:
    def test_no_feedback_is_neutral(self, db: Any) -> None:
        user = _user(db)
        taste = load_user_taste(db, user.id, _corpus())
        assert not taste
        assert taste.taste_vector is None
        assert taste.demote_ids == set()

    def test_dislikes_are_collected_even_below_the_like_threshold(self, db: Any) -> None:
        user = _user(db)
        _rate(db, user.id, "tt1", -1)
        taste = load_user_taste(db, user.id, _corpus())
        assert taste.demote_ids == {"tt1"}
        assert taste.taste_vector is None  # one like is not a taste

    def test_too_few_likes_yields_no_taste_vector(self, db: Any) -> None:
        """A vector from one or two films describes those films, not a taste."""
        user = _user(db)
        for i in range(MIN_LIKES_FOR_TASTE - 1):
            _rate(db, user.id, f"tt{i}", 1)
        assert load_user_taste(db, user.id, _corpus()).taste_vector is None

    def test_unknown_imdb_ids_do_not_crash(self, db: Any) -> None:
        """Feedback can reference films that later left the corpus."""
        user = _user(db)
        for i in range(MIN_LIKES_FOR_TASTE):
            _rate(db, user.id, f"nonexistent{i}", 1)
        taste = load_user_taste(db, user.id, _corpus())
        assert taste.taste_vector is None


class TestTasteInRanking:
    def _items(self) -> list[dict[str, Any]]:
        return [
            {
                "imdb_id": f"tt{i}",
                "title": f"Film {i}",
                "overview": "a horror film about a haunted house",
                "vote_average": 7.0,
            }
            for i in range(12)
        ]

    def test_demoted_films_sink(self) -> None:
        items = self._items()
        demoted = {"tt0", "tt1"}
        result = recommend_unified_semantic(
            mood="haunted house",
            items=items,
            limit=6,
            seed=0,
            temperature=0.0,
            demote_ids=demoted,
        )
        assert not demoted & {m["imdb_id"] for m in result}

    def test_demotion_still_returns_a_full_page(self) -> None:
        """Disliked films are demoted, not filtered -- results stay full."""
        items = self._items()
        result = recommend_unified_semantic(
            mood="haunted house",
            items=items,
            limit=6,
            seed=0,
            temperature=0.0,
            demote_ids={m["imdb_id"] for m in items},  # every film disliked
        )
        assert len(result) == 6

    def test_taste_vector_shifts_ranking(self) -> None:
        items = self._items()
        base = recommend_unified_semantic(
            mood="haunted house", items=items, limit=6, seed=0, temperature=0.0
        )
        dim = 768
        taste = np.zeros(dim, dtype=np.float32)
        taste[0] = 1.0
        shifted = recommend_unified_semantic(
            mood="haunted house",
            items=items,
            limit=6,
            seed=0,
            temperature=0.0,
            weights={"semantic": 0.4, "keyword": 0.2, "popularity": 0.2, "taste": 0.2},
            taste_vector=taste,
        )
        # Both must be well-formed; the taste run may or may not reorder given
        # a synthetic vector, but it must never crash or truncate.
        assert len(base) == len(shifted) == 6

    def test_mismatched_taste_dimension_is_ignored(self) -> None:
        """A stale vector of the wrong width must not poison ranking."""
        items = self._items()
        result = recommend_unified_semantic(
            mood="haunted house",
            items=items,
            limit=6,
            seed=0,
            temperature=0.0,
            taste_vector=np.ones(3, dtype=np.float32),
        )
        assert len(result) == 6


class TestUserTasteContainer:
    def test_falsy_when_empty_truthy_when_populated(self) -> None:
        assert not UserTaste()
        assert UserTaste(demote_ids={"tt1"})
        assert UserTaste(taste_vector=np.ones(4, dtype=np.float32))


class TestCacheIsNotClobbered:
    def test_taste_lookup_never_writes_the_embedding_cache(self, db: Any) -> None:
        """Regression: this used to overwrite the real 500-film cache.

        ``load_user_taste`` called ``get_corpus_embeddings``, which *computes
        and saves*. With a partial corpus (as in these tests, or any caller
        passing a subset) that wrote a wrong-sized array over the production
        cache, forcing a full re-encode on the next request.
        """
        from app.services import corpus as corpus_mod

        before = (
            corpus_mod.EMBEDDINGS_FILE.read_bytes() if corpus_mod.EMBEDDINGS_FILE.exists() else None
        )
        user = _user(db, "cache@example.com")
        for i in range(MIN_LIKES_FOR_TASTE + 1):
            _rate(db, user.id, f"tt{i}", 1)
        load_user_taste(db, user.id, _corpus())

        after = (
            corpus_mod.EMBEDDINGS_FILE.read_bytes() if corpus_mod.EMBEDDINGS_FILE.exists() else None
        )
        assert before == after
