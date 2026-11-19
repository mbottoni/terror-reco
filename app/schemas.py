"""Request and response bodies for the JSON endpoints.

Declared as models so FastAPI rejects a malformed or mistyped body with a 422
that names the field, instead of the handler failing on ``body.get`` and
returning a 500.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints

# Matches the String(20) imdb_id columns.
ImdbId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20)]


class FeedbackIn(BaseModel):
    imdb_id: ImdbId
    rating: Literal[1, -1]
    title: str = ""
    mood: str | None = None
    strategy: str | None = None


class FeedbackOut(BaseModel):
    status: Literal["created", "updated", "removed"]
    imdb_id: str
    rating: Literal[1, 0, -1]


class WatchlistToggleIn(BaseModel):
    imdb_id: ImdbId
    title: str = ""
    poster_url: str | None = None


class WatchlistToggleOut(BaseModel):
    status: Literal["added", "removed"]
    imdb_id: str
    saved: bool
