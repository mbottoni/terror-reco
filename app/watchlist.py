"""Watchlist: films a user saved to watch later.

Kept separate from like/dislike feedback on purpose.  A like says something
about taste in films already seen; saving says only "I intend to watch this".
Feeding saves into the taste vector would describe films the user has no
opinion on yet.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from starlette.responses import Response

from .auth import get_current_user
from .db import get_db
from .models import User, WatchlistItem
from .schemas import WatchlistToggleIn, WatchlistToggleOut
from .security import safe_url

router = APIRouter(prefix="/watchlist")


@router.get("/", response_class=HTMLResponse)
async def watchlist_page(
    request: Request,
    user: User | None = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    if not user:
        return RedirectResponse(url="/auth/login", status_code=303)
    items = (
        db.query(WatchlistItem)
        .filter(WatchlistItem.user_id == user.id)
        .order_by(WatchlistItem.created_at.desc())
        .all()
    )
    resp: Response = request.app.state.templates.TemplateResponse(
        request, "watchlist.html", {"items": items, "user": user}
    )
    return resp


@router.post("/toggle", response_model=WatchlistToggleOut)
async def toggle_watchlist(
    payload: WatchlistToggleIn,
    user: User | None = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Add a film to the watchlist, or remove it if already saved."""
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to save films")

    imdb_id = payload.imdb_id

    existing = (
        db.query(WatchlistItem)
        .filter(WatchlistItem.user_id == user.id, WatchlistItem.imdb_id == imdb_id)
        .first()
    )
    if existing:
        db.delete(existing)
        db.commit()
        return {"status": "removed", "imdb_id": imdb_id, "saved": False}

    db.add(
        WatchlistItem(
            user_id=user.id,
            imdb_id=imdb_id,
            title=payload.title[:512],
            # Straight from a request body and rendered into an <img src> on the
            # watchlist page, so the scheme is checked before it is stored.
            poster_url=safe_url(payload.poster_url)[:1024] or None,
        )
    )
    db.commit()
    return {"status": "added", "imdb_id": imdb_id, "saved": True}
