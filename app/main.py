from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

from .auth import get_current_user
from .auth import router as auth_router
from .db import get_db, init_db
from .history import router as history_router
from .history import save_history
from .models import User
from .services.recommender import recommend_movies, recommend_movies_advanced
from .services.unified_recommender import recommend_unified_semantic
from .settings import get_settings
from .stripe_payments import router as stripe_router

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

settings = get_settings()
app = FastAPI(title=settings.APP_NAME, debug=settings.DEBUG)

_https_only = not settings.DEBUG
print(f"[TerrorReco] DEBUG={settings.DEBUG}, session https_only={_https_only}")

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.SECRET_KEY,
    session_cookie=settings.SESSION_COOKIE_NAME,
    https_only=_https_only,
    same_site="lax",
    max_age=60 * 60 * 24 * 30,
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.state.templates = templates


# Initialize DB on startup (for SQLite dev); for Postgres use migrations
@app.on_event("startup")
async def _startup() -> None:
    init_db()
    # Pre-load the sentence-transformer model so the first request is fast.
    # Wrapped in try/except so the app still starts when the model isn't
    # available (e.g. CI, offline environments).
    try:
        from .services.unified_recommender import _get_sbert

        _get_sbert()
    except Exception:  # noqa: BLE001
        import logging

        logging.getLogger(__name__).warning(
            "Could not pre-load sentence-transformer model at startup; "
            "it will be loaded lazily on the first request."
        )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, user: User | None = Depends(get_current_user)) -> HTMLResponse:
    flash = request.session.pop("flash", None)
    flash_type = request.session.pop("flash_type", "success")
    print(f"[HOME] user_id_in_session={request.session.get('user_id')}, user={user}, flash={flash}")
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "flash": flash, "flash_type": flash_type, "user": user},
    )


@app.get("/loading", response_class=HTMLResponse)
async def loading(request: Request, mood: str = "") -> HTMLResponse:
    return templates.TemplateResponse("loading.html", {"request": request, "mood": mood})


# Valid strategy values accepted by the UI.
STRATEGY_LABELS: dict[str, str] = {
    "keyword": "Keyword Match",
    "embedding": "TF-IDF Similarity",
    "semantic": "Semantic Search",
    "unified": "Unified (AI + Diversity)",
}


@app.get("/recommend", response_class=HTMLResponse)
async def ui_recommendations(
    request: Request,
    mood: str = Query(..., min_length=1),
    strategy: str = Query(default="semantic"),
    min_year: int | None = Query(default=None, ge=1900, le=2100),
    max_year: int | None = Query(default=None, ge=1900, le=2100),
    limit: int = Query(default=6, ge=1, le=20),
    kind: str = Query(default="movie"),  # movie | series | both
    english: int | None = Query(default=None),
    user: User | None = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    strategy_key = strategy if strategy in STRATEGY_LABELS else "semantic"

    if strategy_key == "unified":
        # Full pipeline: corpus semantic search → unified re-ranking with MMR
        pool = await recommend_movies_advanced(
            mood=mood,
            limit=max(limit * 10, 60),
            min_year=min_year,
            max_year=max_year,
            kind=kind,
            english_only=bool(english),
            pages=3,
        )
        movies = recommend_unified_semantic(
            mood=mood,
            items=pool,
            limit=limit,
            diversity_lambda=settings.UNIFIED_DIVERSITY_LAMBDA,
        )
    elif strategy_key == "semantic":
        # Corpus-based sentence-transformer semantic search
        movies = await recommend_movies_advanced(
            mood=mood,
            limit=limit,
            min_year=min_year,
            max_year=max_year,
            kind=kind,
            english_only=bool(english),
            pages=3,
        )
    elif strategy_key == "embedding":
        # TF-IDF cosine similarity on OMDb plot descriptions
        movies = await recommend_movies(
            mood=mood, limit=limit, strategy="embedding"
        )
    else:
        # Keyword: OMDb title search ranked by IMDb rating
        movies = await recommend_movies(
            mood=mood, limit=limit, strategy="keyword"
        )

    # Save history if logged in
    if user:
        save_history(db, user.id, mood, strategy_key, movies)
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "mood": mood,
            "movies": movies,
            "strategy": strategy_key,
            "strategy_label": STRATEGY_LABELS[strategy_key],
        },
    )


@app.get("/api/recommendations")
async def api_recommendations(
    mood: str = Query(..., min_length=1), limit: int = 6
) -> dict[str, Any]:
    try:
        movies = await recommend_movies(mood=mood, limit=limit)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"mood": mood, "count": len(movies), "results": movies}


# Routers
app.include_router(auth_router)
app.include_router(history_router)
app.include_router(stripe_router)
