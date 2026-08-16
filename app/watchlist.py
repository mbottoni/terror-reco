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


@router.post("/toggle")
async def toggle_watchlist(
    request: Request,
    user: User | None = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Add a film to the watchlist, or remove it if already saved.

    Expects JSON ``{"imdb_id": ..., "title": ..., "poster_url": ...}``.
    """
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to save films")

    body = await request.json()
    imdb_id = (body.get("imdb_id") or "").strip()
    if not imdb_id:
        raise HTTPException(status_code=400, detail="imdb_id is required")

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
            title=(body.get("title") or "")[:512],
            poster_url=(body.get("poster_url") or None),
        )
    )
    db.commit()
    return {"status": "added", "imdb_id": imdb_id, "saved": True}
