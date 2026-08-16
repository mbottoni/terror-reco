"""Turn stored like/dislike feedback into a ranking signal.

``MovieFeedback`` rows have been collected since the feature shipped but were
only ever read back to paint the button states -- the signal never reached the
recommender.  This module closes that loop.

Two mechanisms, deliberately different in strength:

* **Taste vector** -- the mean embedding of the films a user liked.  Candidates
  are scored by cosine similarity to it, blended in as one more weighted
  signal.  This is a *soft* preference: it nudges, it does not dictate.
* **Demotion** -- films the user explicitly disliked are pushed below
  everything else.  This is a hard, per-film response to an explicit signal.

Disliked films are demoted rather than filtered out so that a user who has
rated a great deal still receives a full page of results.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

from ..models import MovieFeedback

# A taste vector built from one or two films is mostly noise about that
# specific film rather than a preference, so require a few ratings first.
MIN_LIKES_FOR_TASTE = 3


class UserTaste:
    """A user's feedback, reduced to what the ranker needs."""

    __slots__ = ("taste_vector", "demote_ids", "liked_ids")

    def __init__(
        self,
        taste_vector: np.ndarray | None = None,
        demote_ids: set[str] | None = None,
        liked_ids: set[str] | None = None,
    ) -> None:
        self.taste_vector = taste_vector
        self.demote_ids = demote_ids or set()
        self.liked_ids = liked_ids or set()

    def __bool__(self) -> bool:
        return self.taste_vector is not None or bool(self.demote_ids)


def load_user_taste(db: Session, user_id: int, corpus: list[dict[str, str]]) -> UserTaste:
    """Build a :class:`UserTaste` from a user's stored feedback.

    The taste vector is the mean of the corpus embeddings of liked films.
    Because those embeddings are already L2-normalised and cached, this costs
    a lookup and a mean -- no model inference.
    """
    rows = (
        db.query(MovieFeedback.imdb_id, MovieFeedback.rating)
        .filter(MovieFeedback.user_id == user_id)
        .all()
    )
    if not rows:
        return UserTaste()

    liked = {r.imdb_id for r in rows if r.rating > 0}
    disliked = {r.imdb_id for r in rows if r.rating < 0}

    taste_vector = None
    if len(liked) >= MIN_LIKES_FOR_TASTE:
        from .corpus import load_cached_embeddings

        # Read-only: personalisation must not trigger a 500-film re-encode
        # inside a request, nor write a cache for a partial corpus.
        embeddings = load_cached_embeddings(corpus)  # type: ignore[arg-type]
        idx = [i for i, m in enumerate(corpus) if m.get("imdb_id") in liked]
        if embeddings is not None and idx:
            mean = embeddings[idx].mean(axis=0)
            norm = float(np.linalg.norm(mean))
            # Re-normalise so cosine against the (normalised) item embeddings
            # stays on the same scale as every other signal.
            if norm > 1e-9:
                taste_vector = (mean / norm).astype(np.float32)

    return UserTaste(taste_vector=taste_vector, demote_ids=disliked, liked_ids=liked)
