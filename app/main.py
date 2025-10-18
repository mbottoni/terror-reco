from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from .auth import get_current_user
from .auth import router as auth_router
from .db import get_db, init_db
from .history import router as history_router
from .history import save_history
from .services.recommender import recommend_movies, recommend_movies_advanced
from .settings import get_settings
from .stripe_payments import router as stripe_router

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

settings = get_settings()
app = FastAPI(title=settings.APP_NAME, debug=settings.DEBUG)
app.add_middleware(
	SessionMiddleware,
	secret_key=settings.SECRET_KEY,
	session_cookie=settings.SESSION_COOKIE_NAME,
	https_only=not settings.DEBUG,
	max_age=60 * 60 * 24 * 30,
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.state.templates = templates

# Initialize DB on startup (for SQLite dev); for Postgres use migrations
@app.on_event("startup")
async def _startup() -> None:
	init_db()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, user=Depends(get_current_user)) -> HTMLResponse:
	flash = request.session.pop("flash", None)
	return templates.TemplateResponse("index.html", {"request": request, "flash": flash, "user": user})


@app.get("/loading", response_class=HTMLResponse)
async def loading(request: Request, mood: str = "") -> HTMLResponse:
	return templates.TemplateResponse("loading.html", {"request": request, "mood": mood})


@app.get("/recommend", response_class=HTMLResponse)
async def ui_recommendations(
	request: Request,
	mood: str = Query(..., min_length=1),
	min_year: int | None = Query(default=None, ge=1900, le=2100),
	max_year: int | None = Query(default=None, ge=1900, le=2100),
	limit: int = Query(default=6, ge=1, le=20),
	kind: str = Query(default="movie"),  # movie | series | both
	english: int | None = Query(default=None),
	user=Depends(get_current_user),
	db=Depends(get_db),
) -> HTMLResponse:
	# Use advanced recommender if any advanced filters provided, else fallback
	use_advanced = any([
		min_year is not None,
		max_year is not None,
		kind in ("series", "both"),
		english is not None,
		limit != 6,
	])
	if use_advanced:
		movies = await recommend_movies_advanced(
			mood=mood,
			limit=limit,
			min_year=min_year,
			max_year=max_year,
			kind=kind,
			english_only=bool(english),
			pages=3,
		)
	else:
		movies = await recommend_movies(mood=mood, limit=limit)
	# Save history if logged in
	if user:
		save_history(db, user.id, mood, None, movies)
	return templates.TemplateResponse(
		"results.html",
		{
			"request": request,
			"mood": mood,
			"movies": movies,
		},
	)


@app.get("/api/recommendations")
async def api_recommendations(mood: str = Query(..., min_length=1), limit: int = 6) -> dict[str, Any]:
	try:
		movies = await recommend_movies(mood=mood, limit=limit)
	except HTTPException:
		raise
	except Exception as exc:  # pragma: no cover
		raise HTTPException(status_code=500, detail=str(exc))

	return {"mood": mood, "count": len(movies), "results": movies}


# Routers
app.include_router(auth_router)
app.include_router(history_router)
app.include_router(stripe_router)
