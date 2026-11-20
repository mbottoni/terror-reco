"""Request and response bodies for the JSON endpoints.

Declared as models so FastAPI rejects a malformed or mistyped body with a 422
that names the field, instead of the handler failing on ``body.get`` and
returning a 500.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

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


class SetFeedbackIn(BaseModel):
    mood: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    imdb_ids: Annotated[list[ImdbId], Field(min_length=1, max_length=20)]
    verdict: Literal[1, -1]
    strategy: str | None = None


class SetFeedbackOut(BaseModel):
    status: Literal["created", "updated", "removed"]
    verdict: Literal[1, 0, -1]


class InteractionIn(BaseModel):
    result_key: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    imdb_id: ImdbId
    kind: Literal["open", "imdb_click", "save", "similar_click"]


class InteractionOut(BaseModel):
    status: Literal["recorded", "duplicate", "ignored"]


MIN_DONATION = 1
MAX_DONATION = 500


class CheckoutIn(BaseModel):
    """Optional body for starting a donation. No body means the default amount."""

    # Whole units of the donation currency. The bounds are deliberate: under
    # one unit Stripe's fee eats the donation, and the cap stops a typo.
    amount: Annotated[int, Field(ge=MIN_DONATION, le=MAX_DONATION)] | None = None
