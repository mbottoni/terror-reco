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
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    history: Mapped[list[SearchHistory]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    feedback: Mapped[list[MovieFeedback]] = relationship(
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
