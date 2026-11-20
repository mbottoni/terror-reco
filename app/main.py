import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import Response

from .account import router as account_router
from .auth import get_current_user
from .auth import router as auth_router
from .db import get_db, init_db
from .history import router as history_router
from .history import save_history
from .middleware import RateLimitMiddleware, SecurityHeadersMiddleware
from .models import MovieFeedback, User, WatchlistItem
from .schemas import FeedbackIn, FeedbackOut
from .security import poster_variant, require_csrf, safe_url, session_csrf_token
from .services.corpus import CorpusNotBuiltError
from .services.personalization import UserTaste, load_user_taste
from .services.recommender import (
    explain_match,
    films_by_ids,
    recommend_movies_advanced,
    similar_movies,
)
from .services.unified_recommender import DEFAULT_WEIGHTS, recommend_unified_semantic
from .settings import get_settings
from .static_assets import CachedStaticFiles, static_url
from .stripe_payments import router as stripe_router
from .watchlist import router as watchlist_router

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

settings = get_settings()


def _warm_model_and_corpus() -> None:
    """Load the model and the corpus before the first request needs them.

    Every step is best-effort: booting without the model (CI, offline) or
    without a corpus is deliberate, and /healthz reports the latter.
    """
    log = logging.getLogger(__name__)
    try:
        from .services.unified_recommender import _get_sbert

        _get_sbert()
    except Exception:  # noqa: BLE001
        log.warning(
            "Could not pre-load sentence-transformer model at startup; "
            "it will be loaded lazily on the first request."
        )
    try:
        from .services.corpus import (
            get_corpus_and_embeddings,
            load_cached_embeddings,
            load_corpus,
        )

        # Only warm from a valid on-disk cache. Without one,
        # get_corpus_and_embeddings() would re-encode the whole corpus here
        # and hold up startup; that case stays lazy.
        if load_cached_embeddings(load_corpus()) is not None:
            corpus, _ = get_corpus_and_embeddings()
            log.info("Corpus warmed: %d films", len(corpus))
    except Exception:  # noqa: BLE001
        log.warning("Could not warm the corpus at startup", exc_info=True)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    init_db()
    _warm_model_and_corpus()
    yield


app = FastAPI(title=settings.APP_NAME, debug=settings.DEBUG, lifespan=lifespan)

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
# Added after the session middleware, so these wrap it: a rate-limited request
# is refused before any session or database work, and every response --
# including that refusal -- carries the security headers.
if settings.RATE_LIMIT_ENABLED:
    app.add_middleware(RateLimitMiddleware)
app.add_middleware(SecurityHeadersMiddleware, hsts=_https_only)
app.mount("/static", CachedStaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
# Autoescaping makes a value safe as text but says nothing about a URL scheme,
# so every poster URL rendered into a src/href goes through this first.
templates.env.filters["safe_url"] = safe_url
templates.env.globals["static_url"] = static_url
templates.env.filters["poster_variant"] = poster_variant
# Every page embeds the session's CSRF token (see _base.html), so forms and
# fetch() calls on any page can send it back.
templates.env.globals["csrf_token"] = session_csrf_token
app.state.templates = templates


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


def refine_context(
    *,
    mood: str,
    strategy: str,
    min_year: int | None,
    max_year: int | None,
    min_rating: float | None,
    limit: int,
    english: bool,
) -> dict[str, Any]:
    """What the results page needs to adjust a search without starting over.

    ``params`` are the active, non-default query parameters (never ``seed``:
    refining wants a fresh draw). Each chip names one active filter and links
    to the same search without it. ``shuffle_url`` repeats the search as is.
    """
    from urllib.parse import urlencode

    params: dict[str, str] = {"mood": mood, "strategy": strategy}
    labels: dict[str, str] = {}
    if min_year is not None:
        params["min_year"] = str(min_year)
        labels["min_year"] = f"From {min_year}"
    if max_year is not None:
        params["max_year"] = str(max_year)
        labels["max_year"] = f"Up to {max_year}"
    if min_rating is not None:
        params["min_rating"] = f"{min_rating:g}"
        labels["min_rating"] = f"IMDb {min_rating:g}+"
    if english:
        params["english"] = "1"
        labels["english"] = "English only"
    if limit != DEFAULT_RESULT_LIMIT:
        params["limit"] = str(limit)
        labels["limit"] = f"{limit} results"

    chips = [
        {
            "label": label,
            "remove_url": "/recommend?" + urlencode({k: v for k, v in params.items() if k != key}),
        }
        for key, label in labels.items()
    ]
    return {
        "params": {k: v for k, v in params.items() if k != "mood"},
        "chips": chips,
        "shuffle_url": "/recommend?" + urlencode(params),
    }


DEFAULT_RESULT_LIMIT = 6

# Valid strategy values accepted by the UI. Both search the local corpus; an
# unknown value (including the retired "keyword") falls back to the default.
STRATEGY_LABELS: dict[str, str] = {
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
    limit: int = Query(default=DEFAULT_RESULT_LIMIT, ge=1, le=20),
    english: int | None = Query(default=None),
    seed: int | None = Query(
        default=None, description="reproducible results; also disables sampling noise"
    ),
    ids: str | None = Query(
        default=None,
        max_length=400,
        description="comma-separated IMDb ids: show exactly these films (shared links)",
    ),
    user: User | None = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    default_strategy = "unified" if settings.USE_UNIFIED_RECOMMENDER else "semantic"
    strategy_key = strategy if strategy in STRATEGY_LABELS else default_strategy

    # A shared link names the films themselves. A seed cannot do this job:
    # the set on screen is a random draw, possibly personalised, and replaying
    # a seed gives the recipient a different one.
    shared = await run_in_threadpool(films_by_ids, ids.split(",")) if ids else []

    if shared:
        movies = shared
    elif strategy_key == "unified":
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
    else:
        # Corpus-based sentence-transformer semantic search
        movies = await run_in_threadpool(
            recommend_movies_advanced,
            mood=mood,
            limit=limit,
            min_year=min_year,
            max_year=max_year,
            min_rating=min_rating,
            english_only=bool(english),
            seed=seed,
            temperature=0.0 if seed is not None else 1.0,
        )

    # Explain each result in the user's own words: which of the film's
    # keywords the query actually hit.
    for movie in movies:
        movie["match_terms"] = explain_match(mood, movie)

    # Save history if logged in. Opening someone's shared set is not a search
    # the viewer made, so it is not recorded as one.
    if user and not shared:
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
            "strategy_labels": STRATEGY_LABELS,
            "is_shared": bool(shared),
            "refine": refine_context(
                mood=mood,
                strategy=strategy_key,
                min_year=min_year,
                max_year=max_year,
                min_rating=min_rating,
                limit=limit,
                english=bool(english),
            ),
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
        # Uses the same corpus pipeline as the UI.
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


@app.post("/api/feedback", response_model=FeedbackOut, dependencies=[Depends(require_csrf)])
async def submit_feedback(
    payload: FeedbackIn,
    user: User | None = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Record a like (+1) or dislike (-1) for a movie.

    Upserts: if the user already rated this movie, the rating is updated.
    Sending the same rating again removes the feedback (toggle off).
    """
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to rate movies")

    imdb_id = payload.imdb_id
    rating: int = payload.rating
    # Free text is clipped to the column widths rather than rejected: a long
    # mood is a legitimate query, and Postgres would refuse the overflow.
    title = payload.title[:512]
    mood = payload.mood[:512] if payload.mood else payload.mood
    strategy = payload.strategy[:64] if payload.strategy else payload.strategy

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
app.include_router(account_router)
app.include_router(history_router)
app.include_router(stripe_router)
app.include_router(watchlist_router)
