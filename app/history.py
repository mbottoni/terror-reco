from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from starlette.responses import Response

from .auth import get_current_user
from .db import get_db
from .models import SearchHistory, SearchResult, User

router = APIRouter(prefix="/history")


@router.get("/", response_class=HTMLResponse)
async def history_page(
    request: Request, user: User | None = Depends(get_current_user), db: Session = Depends(get_db)
) -> Response:
    if not user:
        return RedirectResponse(url="/auth/login", status_code=303)
    items = (
        db.query(SearchHistory)
        .filter(SearchHistory.user_id == user.id)
        .order_by(SearchHistory.created_at.desc())
        .limit(50)
        .all()
    )
    resp: Response = request.app.state.templates.TemplateResponse(
        request, "history.html", {"items": items}
    )
    return resp


def save_history(
    db: Session, user_id: int, mood: str, strategy: str | None, results: list[dict[str, Any]]
) -> None:
    """Record a search and the films it returned, in rank order.

    Stores film *references* rather than the full dicts that used to go into
    ``results_json``: the corpus already holds the details, and references are
    what make the history queryable.
    """
    entry = SearchHistory(user_id=user_id, mood=mood, strategy=strategy)
    entry.results = [
        SearchResult(
            position=i,
            imdb_id=(movie.get("imdb_id") or "")[:20],
            title=(movie.get("title") or "")[:512],
        )
        for i, movie in enumerate(results)
        if movie.get("imdb_id")
    ]
    db.add(entry)
    db.commit()
