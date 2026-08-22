import logging
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import Response

from .auth import get_current_user
from .auth import router as auth_router
from .db import get_db, init_db
from .history import router as history_router
from .history import save_history
from .models import MovieFeedback, User, WatchlistItem
from .security import safe_url
from .services.corpus import CorpusNotBuiltError
from .services.personalization import UserTaste, load_user_taste
from .services.recommender import (
    explain_match,
    recommend_movies,
    recommend_movies_advanced,
    similar_movies,
)
from .services.unified_recommender import DEFAULT_WEIGHTS, recommend_unified_semantic
from .settings import get_settings
from .stripe_payments import router as stripe_router
from .watchlist import router as watchlist_router

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

settings = get_settings()
app = FastAPI(title=settings.APP_NAME, debug=settings.DEBUG)

_https_only = not settings.DEBUG
logger = logging.getLogger(__name__)
logger.info("DEBUG=%s, session https_only=%s", settings.DEBUG, _https_only)

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
# Autoescaping makes a value safe as text but says nothing about a URL scheme,
# so every poster URL rendered into a src/href goes through this first.
templates.env.filters["safe_url"] = safe_url
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


def _wants_json(request: Request) -> bool:
    """True for the JSON API, which must keep answering in JSON."""
    return request.url.path.startswith("/api/")


def _error_page(
    request: Request, *, status_code: int, icon: str, heading: str, message: str
) -> Response:
    resp: Response = templates.TemplateResponse(
        request,
        "error.html",
        {"icon": icon, "heading": heading, "message": message, "user": None},
        status_code=status_code,
    )
    return resp


@app.exception_handler(CorpusNotBuiltError)
async def _corpus_not_built_handler(request: Request, exc: Exception) -> Response:
    """A missing corpus is a deployment problem, not a bad query.

    ``/api/recommendations`` already turned this into a clean 503, but the HTML
    route let it escape as a 500 traceback -- so the same server-side condition
    looked like two different failures depending on which door you came in.
    ``/healthz`` reports it as ``degraded``; this makes the user-facing pages
    agree with the probe.
    """
    logger.error("Recommendation requested with no corpus: %s", exc)
    if _wants_json(request):
        return JSONResponse(status_code=503, content={"detail": str(exc)})
    return _error_page(
        request,
        status_code=503,
        icon="\U0001f9ea",
        heading="The film corpus is not available",
        message=(
            "Nothing is wrong with your search -- this server has no film data loaded "
            "yet. It is a deployment problem on our side, not something you can fix by "
            "trying a different mood."
        ),
    )


@app.exception_handler(StarletteHTTPException)
async def _http_exception_handler(request: Request, exc: Exception) -> Response:
    """Render 404s as a page; everything else keeps its default behaviour."""
    assert isinstance(exc, StarletteHTTPException)
    if exc.status_code == 404 and not _wants_json(request):
        return _error_page(
            request,
            status_code=404,
            icon="\U0001f573\ufe0f",
            heading="Page not found",
            message="That page does not exist. It may have been moved or never existed.",
        )
    return await http_exception_handler(request, exc)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> Response:
    """Last resort: an HTML page instead of a bare plain-text 500."""
    logger.exception("Unhandled error serving %s", request.url.path)
    if _wants_json(request):
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
    return _error_page(
        request,
        status_code=500,
        icon="\U0001f480",
        heading="Something went wrong",
        message="An unexpected error occurred. It has been logged.",
    )


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    """Liveness/readiness probe.

    Reports corpus availability so a deploy that shipped without one is
    visible from the platform rather than only from a failing user request.
    """
    from .services.corpus import get_corpus_and_embeddings

    try:
        corpus, embeddings = get_corpus_and_embeddings()
        corpus_ok = bool(corpus) and embeddings.shape[0] == len(corpus)
        return {
            "status": "ok" if corpus_ok else "degraded",
            "corpus_films": len(corpus),
            "embeddings_ready": corpus_ok,
        }
    except Exception as exc:  # noqa: BLE001 - a probe must always answer
        return {"status": "degraded", "corpus_films": 0, "error": str(exc)}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, user: User | None = Depends(get_current_user)) -> HTMLResponse:
    flash = request.session.pop("flash", None)
    flash_type = request.session.pop("flash_type", "success")
    return templates.TemplateResponse(
        request,
        "index.html",
        {"flash": flash, "flash_type": flash_type, "user": user},
    )


@app.get("/loading", response_class=HTMLResponse)
async def loading(request: Request, mood: str = "") -> HTMLResponse:
    return templates.TemplateResponse(request, "loading.html", {"mood": mood})


def _load_taste(db: Session, user: User | None) -> UserTaste:
    """Look up a signed-in user's taste, degrading to neutral on any failure."""
    if not user or not settings.PERSONALIZATION_ENABLED:
        return UserTaste()
    try:
        from .services.corpus import load_corpus

        return load_user_taste(db, user.id, load_corpus())
    except Exception:  # noqa: BLE001 - personalisation must never break search
        logging.getLogger(__name__).warning("Could not load user taste", exc_info=True)
        return UserTaste()


# Valid strategy values accepted by the UI.
STRATEGY_LABELS: dict[str, str] = {
    "keyword": "Keyword Match",
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
    min_rating: float | None = Query(default=None, ge=0, le=10),
    limit: int = Query(default=6, ge=1, le=20),
    kind: str = Query(default="movie"),  # movie | series | both
    english: int | None = Query(default=None),
    seed: int | None = Query(
        default=None, description="reproducible results; also disables sampling noise"
    ),
    user: User | None = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    default_strategy = "unified" if settings.USE_UNIFIED_RECOMMENDER else "semantic"
    strategy_key = strategy if strategy in STRATEGY_LABELS else default_strategy

    if strategy_key == "unified":
        # Full pipeline: corpus semantic search → unified re-ranking with MMR
        # Both stages below are CPU-bound (sbert forward pass + numpy) and do
        # no I/O, so they run in a worker thread; on the event loop they would
        # block every other request for the duration.
        pool = await run_in_threadpool(
            recommend_movies_advanced,
            mood=mood,
            limit=max(limit * 10, 60),
            min_year=min_year,
            max_year=max_year,
            min_rating=min_rating,
            kind=kind,
            english_only=bool(english),
            seed=seed,
            temperature=0.0 if seed is not None else 1.0,
            keep_internal=True,  # preserves _embedding_row so unified reuses cached vectors
        )
        taste = _load_taste(db, user)
        weights = dict(settings.UNIFIED_WEIGHTS) if settings.UNIFIED_WEIGHTS else None
        if taste.taste_vector is not None:
            # Blend the taste signal in without renormalising the others: it
            # is an additive nudge, not a replacement for relevance.
            weights = dict(weights or DEFAULT_WEIGHTS)
            weights["taste"] = settings.PERSONALIZATION_TASTE_WEIGHT

        movies = await run_in_threadpool(
            recommend_unified_semantic,
            mood=mood,
            items=pool,
            limit=limit,
            diversity_lambda=settings.UNIFIED_DIVERSITY_LAMBDA,
            weights=weights,
            seed=seed,
            temperature=0.0 if seed is not None else 1.0,
            taste_vector=taste.taste_vector,
            demote_ids=taste.demote_ids,
            use_cross_encoder=settings.UNIFIED_USE_CROSS_ENCODER,
        )
    elif strategy_key == "semantic":
        # Corpus-based sentence-transformer semantic search
        movies = await run_in_threadpool(
            recommend_movies_advanced,
            mood=mood,
            limit=limit,
            min_year=min_year,
            max_year=max_year,
            min_rating=min_rating,
            kind=kind,
            english_only=bool(english),
            seed=seed,
            temperature=0.0 if seed is not None else 1.0,
        )
    else:
        # Keyword: OMDb title search ranked by IMDb rating
        movies = await recommend_movies(mood=mood, limit=limit, strategy="keyword")

    # Explain each result in the user's own words: which of the film's
    # keywords the query actually hit.
    for movie in movies:
        movie["match_terms"] = explain_match(mood, movie)

    # Save history if logged in
    if user:
        save_history(db, user.id, mood, strategy_key, movies)
    # Load existing feedback for this user so the UI can show active states
    user_feedback: dict[str, int] = {}
    if user:
        rows = (
            db.query(MovieFeedback.imdb_id, MovieFeedback.rating)
            .filter(MovieFeedback.user_id == user.id)
            .all()
        )
        user_feedback = {r.imdb_id: r.rating for r in rows}

    saved_ids: list[str] = []
    if user:
        saved_ids = [
            row.imdb_id
            for row in db.query(WatchlistItem.imdb_id).filter(WatchlistItem.user_id == user.id)
        ]

    return templates.TemplateResponse(
        request,
        "results.html",
        {
            "mood": mood,
            "saved_ids": saved_ids,
            "movies": movies,
            "strategy": strategy_key,
            "strategy_label": STRATEGY_LABELS[strategy_key],
            "user": user,
            "user_feedback": user_feedback,
        },
    )


@app.get("/api/recommendations")
async def api_recommendations(
    mood: str = Query(..., min_length=1),
    limit: int = 6,
    seed: int | None = Query(default=None),
) -> dict[str, Any]:
    try:
        # Uses the same corpus pipeline as the UI. This previously called
        # recommend_movies(), which defaults to the *keyword* strategy, so the
        # JSON API returned materially worse results than the web page for the
        # very same query.
        movies = await run_in_threadpool(
            recommend_movies_advanced, mood=mood, limit=limit, seed=seed
        )
    except CorpusNotBuiltError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"mood": mood, "count": len(movies), "results": movies}


@app.get("/api/similar/{imdb_id}")
async def api_similar(imdb_id: str, limit: int = Query(default=6, ge=1, le=20)) -> dict[str, Any]:
    """Films most like a given one ("more like this").

    Pure item-to-item cosine over cached embeddings -- no query encoding, so
    this is a single dot product rather than a model forward pass.
    """
    movies = await run_in_threadpool(similar_movies, imdb_id=imdb_id, limit=limit)
    if not movies:
        raise HTTPException(status_code=404, detail="Unknown film, or corpus not built")
    return {"imdb_id": imdb_id, "count": len(movies), "results": movies}


@app.post("/api/feedback")
async def submit_feedback(
    request: Request,
    user: User | None = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Record a like (+1) or dislike (-1) for a movie.

    Expects JSON: ``{"imdb_id": "...", "title": "...", "rating": 1,
    "mood": "...", "strategy": "..."}``.
    Upserts: if the user already rated this movie, the rating is updated.
    Sending the same rating again removes the feedback (toggle off).
    """
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to rate movies")

    body = await request.json()
    imdb_id: str = body.get("imdb_id", "")
    title: str = body.get("title", "")
    rating: int = int(body.get("rating", 0))
    mood: str | None = body.get("mood")
    strategy: str | None = body.get("strategy")

    if not imdb_id or rating not in (1, -1):
        raise HTTPException(status_code=400, detail="Invalid feedback payload")

    existing = (
        db.query(MovieFeedback)
        .filter(MovieFeedback.user_id == user.id, MovieFeedback.imdb_id == imdb_id)
        .first()
    )

    if existing:
        if existing.rating == rating:
            # Toggle off: same button pressed again -> remove feedback
            db.delete(existing)
            db.commit()
            return {"status": "removed", "imdb_id": imdb_id, "rating": 0}
        # Switch rating
        existing.rating = rating
        existing.mood = mood
        existing.strategy = strategy
        db.commit()
        return {"status": "updated", "imdb_id": imdb_id, "rating": rating}

    fb = MovieFeedback(
        user_id=user.id,
        imdb_id=imdb_id,
        title=title,
        rating=rating,
        mood=mood,
        strategy=strategy,
    )
    db.add(fb)
    db.commit()
    return {"status": "created", "imdb_id": imdb_id, "rating": rating}


# Routers
app.include_router(auth_router)
app.include_router(history_router)
app.include_router(stripe_router)
app.include_router(watchlist_router)
