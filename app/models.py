from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    history: Mapped[list[SearchHistory]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    feedback: Mapped[list[MovieFeedback]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class SearchHistory(Base):
    __tablename__ = "search_history"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    mood: Mapped[str] = mapped_column(String(512))
    strategy: Mapped[str | None] = mapped_column(String(64), nullable=True)
    results_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, index=True)

    user: Mapped[User] = relationship(back_populates="history")


class MovieFeedback(Base):
    """Stores like/dislike feedback per user + movie.

    Each user can have at most one feedback entry per IMDb ID.
    ``rating`` is +1 (like) or -1 (dislike).  The ``mood`` column
    records the query that produced the recommendation so the signal
    can later be used to personalise results per user per context.
    """

    __tablename__ = "movie_feedback"
    __table_args__ = (
        UniqueConstraint("user_id", "imdb_id", name="uq_user_movie"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    imdb_id: Mapped[str] = mapped_column(String(20), index=True)
    title: Mapped[str] = mapped_column(String(512))
    rating: Mapped[int] = mapped_column()  # +1 = like, -1 = dislike
    mood: Mapped[str | None] = mapped_column(String(512), nullable=True)
    strategy: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow, onupdate=datetime.utcnow
    )

    user: Mapped[User] = relationship(back_populates="feedback")
