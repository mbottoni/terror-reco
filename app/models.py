from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def _utcnow() -> datetime:
    """Timezone-aware UTC now.

    ``datetime.utcnow()`` is deprecated and returns a *naive* datetime, which
    compares incorrectly against aware values and loses the zone on round-trip.
    """
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    # Copied into the session cookie at sign-in and compared on every request.
    # Bumping it invalidates every cookie issued before, which is the only
    # way to revoke a signed-cookie session short of rotating SECRET_KEY.
    session_version: Mapped[int] = mapped_column(default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    history: Mapped[list[SearchHistory]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    feedback: Mapped[list[MovieFeedback]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    watchlist: Mapped[list[WatchlistItem]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class SearchHistory(Base):
    __tablename__ = "search_history"
    # The history page filters by user_id and orders by created_at desc; the
    # two single-column indexes cannot serve that as well as one composite.
    __table_args__ = (Index("ix_search_history_user_created", "user_id", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    mood: Mapped[str] = mapped_column(String(512))
    strategy: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Legacy denormalised blob. No longer written; kept so existing rows
    # remain readable until a follow-up migration drops it.
    results_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )

    user: Mapped[User] = relationship(back_populates="history")
    results: Mapped[list[SearchResult]] = relationship(
        back_populates="search", cascade="all, delete-orphan", order_by="SearchResult.position"
    )


class SearchResult(Base):
    """One film returned by one search, in rank order.

    Replaces stuffing full movie dicts into ``SearchHistory.results_json``
    (~6 KB a row of denormalised plot/cast/poster data). Storing the *reference*
    instead makes the data answerable: which films get recommended most, which
    moods return nothing, whether one strategy outperforms another. The film
    details themselves live in the corpus, which is already the source of truth.
    """

    __tablename__ = "search_results"
    __table_args__ = (Index("ix_search_results_imdb", "imdb_id"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    search_id: Mapped[int] = mapped_column(
        ForeignKey("search_history.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column()  # 0-based rank within the result set
    imdb_id: Mapped[str] = mapped_column(String(20))
    title: Mapped[str] = mapped_column(String(512))

    search: Mapped[SearchHistory] = relationship(back_populates="results")


class WatchlistItem(Base):
    """A film a user saved to watch later.

    Distinct from :class:`MovieFeedback`: a like is a *taste* signal about a
    film already seen, whereas saving is an intent to watch. Conflating them
    would poison the taste vector with films the user has no opinion on yet.
    """

    __tablename__ = "watchlist"
    __table_args__ = (UniqueConstraint("user_id", "imdb_id", name="uq_watchlist_user_movie"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    imdb_id: Mapped[str] = mapped_column(String(20), index=True)
    title: Mapped[str] = mapped_column(String(512))
    poster_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    user: Mapped[User] = relationship(back_populates="watchlist")


class MovieFeedback(Base):
    """Stores like/dislike feedback per user + movie.

    Each user can have at most one feedback entry per IMDb ID.
    ``rating`` is +1 (like) or -1 (dislike).  The ``mood`` column
    records the query that produced the recommendation so the signal
    can later be used to personalise results per user per context.
    """

    __tablename__ = "movie_feedback"
    __table_args__ = (UniqueConstraint("user_id", "imdb_id", name="uq_user_movie"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    imdb_id: Mapped[str] = mapped_column(String(20), index=True)
    title: Mapped[str] = mapped_column(String(512))
    rating: Mapped[int] = mapped_column()  # +1 = like, -1 = dislike
    mood: Mapped[str | None] = mapped_column(String(512), nullable=True)
    strategy: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    user: Mapped[User] = relationship(back_populates="feedback")


class Payment(Base):
    """One completed Stripe checkout.

    Written by the webhook, which is the only trustworthy record that money
    moved: the success page is just a redirect the browser may never follow.
    Stripe retries deliveries, so ``checkout_session_id`` is unique and a
    repeat of the same event changes nothing.
    """

    __tablename__ = "payments"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    checkout_session_id: Mapped[str] = mapped_column(String(255), unique=True)
    stripe_event_id: Mapped[str] = mapped_column(String(255))
    payment_intent_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    amount_total: Mapped[int] = mapped_column()  # minor units (cents)
    currency: Mapped[str] = mapped_column(String(8))
    status: Mapped[str] = mapped_column(String(32))  # "paid" | "refunded"
    # Nullable: most supporters are not signed in, and deleting an account
    # must not delete the record that a payment happened.
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class SetFeedback(Base):
    """A verdict on a whole result set: "good picks" or not.

    Per-film likes need the modal opened and a judgement made, so almost
    nobody leaves them. One click on the page is cheap, and the set is the
    thing the eval actually scores (NDCG@6 is a set-level metric).

    Self-contained on purpose: it stores the mood and the films that were on
    screen, so it works for signed-out visitors, who have no SearchHistory
    row to point at. ``visitor_id`` is a random per-session value, used only
    so one visitor's repeat clicks update their verdict instead of stacking.
    """

    __tablename__ = "set_feedback"
    __table_args__ = (
        UniqueConstraint("visitor_id", "result_key", name="uq_set_feedback_visitor_result"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    visitor_id: Mapped[str] = mapped_column(String(32), index=True)
    # sha256 of mood + ordered film ids: identifies "this set for this mood".
    result_key: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    mood: Mapped[str] = mapped_column(String(512))
    strategy: Mapped[str | None] = mapped_column(String(64), nullable=True)
    imdb_ids: Mapped[list[str]] = mapped_column(JSON)  # in the order shown
    verdict: Mapped[int] = mapped_column()  # +1 = good picks, -1 = not
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
