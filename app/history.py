from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from .auth import get_current_user
from .db import get_db_session
from .models import SearchHistory, User


router = APIRouter(prefix="/history")


@router.get("/", response_class=HTMLResponse)
async def history_page(request: Request, user: Optional[User] = Depends(get_current_user), db: Session = Depends(get_db_session)) -> HTMLResponse:
	if not user:
		raise HTTPException(status_code=401, detail="Login required")
	items = (
		db.query(SearchHistory)
		.filter(SearchHistory.user_id == user.id)
		.order_by(SearchHistory.created_at.desc())
		.limit(50)
		.all()
	)
	return request.app.state.templates.TemplateResponse("history.html", {"request": request, "items": items})


def save_history(db: Session, user_id: int, mood: str, strategy: Optional[str], results: List[Dict[str, Any]]) -> None:
	entry = SearchHistory(user_id=user_id, mood=mood, strategy=strategy, results_json={"results": results})
	db.add(entry)
	db.commit()
