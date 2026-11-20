"""Watchlist: films a user saved to watch later.

Kept separate from like/dislike feedback on purpose.  A like says something
about taste in films already seen; saving says only "I intend to watch this".
Feeding saves into the taste vector would describe films the user has no
opinion on yet.
"""

from __future__ import annotations

import re
import secrets
from typing import Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from starlette.responses import Response

from .auth import get_current_user
from .db import get_db
from .models import PublicList, User, WatchlistItem
from .schemas import WatchlistToggleIn, WatchlistToggleOut
from .security import require_csrf, safe_url

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
        request,
        "watchlist.html",
        {
            "items": items,
            "user": user,
            "public_list": db.query(PublicList).filter(PublicList.user_id == user.id).first(),
            "flash": request.session.pop("flash", None),
            "flash_type": request.session.pop("flash_type", "success"),
        },
    )
    return resp


@router.post("/toggle", response_model=WatchlistToggleOut, dependencies=[Depends(require_csrf)])
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


# ----------------------------------------------------------------- publishing
TITLE_MAX = 80
DESCRIPTION_MAX = 280


def _clean(text: str, limit: int) -> str:
    """One line of plain text: control characters out, whitespace collapsed."""
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _flash(request: Request, message: str, kind: str = "success") -> RedirectResponse:
    request.session["flash"] = message
    request.session["flash_type"] = kind
    return RedirectResponse(url="/watchlist/", status_code=303)


@router.post("/publish", dependencies=[Depends(require_csrf)])
async def publish_list(
    request: Request,
    title: str = Form(...),
    description: str = Form(""),
    user: User | None = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> RedirectResponse:
    """Publish the watchlist, or update the title and description of a published one."""
    if not user:
        return RedirectResponse(url="/auth/login", status_code=303)
    clean_title = _clean(title, TITLE_MAX)
    if not clean_title:
        return _flash(request, "Give your list a title before publishing it.", "error")
    if not db.query(WatchlistItem.id).filter(WatchlistItem.user_id == user.id).first():
        return _flash(request, "Save at least one film before publishing your list.", "error")

    existing = db.query(PublicList).filter(PublicList.user_id == user.id).first()
    if existing:
        existing.title = clean_title
        existing.description = _clean(description, DESCRIPTION_MAX) or None
        db.commit()
        return _flash(request, "Your public list was updated.")
    db.add(
        PublicList(
            user_id=user.id,
            slug=secrets.token_urlsafe(9),
            title=clean_title,
            description=_clean(description, DESCRIPTION_MAX) or None,
        )
    )
    db.commit()
    return _flash(request, "Your list is public. Anyone with the link can see it.")


@router.post("/unpublish", dependencies=[Depends(require_csrf)])
async def unpublish_list(
    request: Request,
    user: User | None = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> RedirectResponse:
    """Withdraw the list. Its address stops working for good."""
    if not user:
        return RedirectResponse(url="/auth/login", status_code=303)
    db.query(PublicList).filter(PublicList.user_id == user.id).delete()
    db.commit()
    return _flash(request, "Your list is private again. The old link no longer works.")


public_router = APIRouter()


@public_router.get("/l/{slug}", response_class=HTMLResponse, response_model=None)
async def public_list_page(request: Request, slug: str, db: Session = Depends(get_db)) -> Response:
    """A published list, read-only, for anyone with the link.

    Shows the films and the title the owner wrote, and nothing that identifies
    the owner. The response asks search engines not to index it: a link the
    owner chose to hand out is not consent to being listed.
    """
    published = db.query(PublicList).filter(PublicList.slug == slug).first()
    if published is None:
        raise HTTPException(status_code=404, detail="List not found")
    items = (
        db.query(WatchlistItem)
        .filter(WatchlistItem.user_id == published.user_id)
        .order_by(WatchlistItem.created_at.desc())
        .all()
    )
    from .services.corpus import get_corpus_and_embeddings

    try:
        by_id = {m.get("imdb_id"): m for m in get_corpus_and_embeddings()[0]}
    except Exception:  # noqa: BLE001 - the list still renders from what was saved
        by_id = {}
    films = [
        {
            "imdb_id": item.imdb_id,
            # Prefer the corpus record; fall back to what was stored when the
            # film was saved, so a corpus rebuild cannot blank a public page.
            "title": (by_id.get(item.imdb_id) or {}).get("title") or item.title,
            "year": (by_id.get(item.imdb_id) or {}).get("year"),
            "poster_url": (by_id.get(item.imdb_id) or {}).get("poster_url") or item.poster_url,
        }
        for item in items
    ]
    page: Response = request.app.state.templates.TemplateResponse(
        request, "public_list.html", {"list": published, "films": films, "user": None}
    )
    page.headers["X-Robots-Tag"] = "noindex, nofollow"
    return page
