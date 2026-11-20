import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BeforeValidator, Field
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import Response

from .account import router as account_router
from .auth import get_current_user
from .auth import router as auth_router
from .db import get_db, init_db
from .email_verification import router as email_verification_router
from .google_auth import google_enabled
from .google_auth import router as google_router
from .history import router as history_router
from .history import save_history
from .middleware import RateLimitMiddleware, SecurityHeadersMiddleware
from .models import Impression, Interaction, MovieFeedback, SetFeedback, User, WatchlistItem
from .observability import configure_error_tracking, configure_logging
from .onboarding import router as onboarding_router
from .password_reset import router as password_reset_router
from .schemas import (
    FeedbackIn,
    FeedbackOut,
    InteractionIn,
    InteractionOut,
    SetFeedbackIn,
    SetFeedbackOut,
)
from .security import poster_variant, require_csrf, safe_url, session_csrf_token
from .services.corpus import CorpusNotBuiltError
from .services.personalization import UserTaste, explain_by_liked_films, load_user_taste
from .services.recommender import (
    explain_match,
    films_by_ids,
    films_without_each_filter,
    recommend_movies_advanced,
    similar_movies,
)
from .services.unified_recommender import DEFAULT_WEIGHTS, recommend_unified_semantic
from .settings import get_settings
from .static_assets import CachedStaticFiles, static_url
from .stripe_payments import router as stripe_router
from .watchlist import public_router as public_list_router
from .watchlist import router as watchlist_router

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

settings = get_settings()
# Before anything logs: without this the app's INFO lines were dropped.
configure_logging(settings.LOG_LEVEL)
configure_error_tracking(settings)


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


def public_url(request: Request, path: str) -> str:
    """Absolute URL for *path*: link previews need them, relative ones fail.

    Built from PUBLIC_BASE_URL (always set in production); in development the
    request's own origin stands in.
    """
    base = settings.PUBLIC_BASE_URL or str(request.base_url).rstrip("/")
    return f"{base}{path}"


def page_url(request: Request) -> str:
    """This page's canonical absolute URL, query string included."""
    query = f"?{request.url.query}" if request.url.query else ""
    return public_url(request, f"{request.url.path}{query}")


templates.env.globals["public_url"] = public_url
templates.env.globals["page_url"] = page_url

# Link-preview fetchers and crawlers. They load result pages without anyone
# looking at them, so they must not count as a film having been "shown".
_BOT_MARKERS = (
    "bot",
    "crawler",
    "spider",
    "preview",
    "facebookexternalhit",
    "slurp",
    "embedly",
    "whatsapp",
    "curl/",
    "python-requests",
    "httpx",
)


def looks_like_bot(request: Request) -> bool:
    agent = request.headers.get("user-agent", "").lower()
    return not agent or any(marker in agent for marker in _BOT_MARKERS)


templates.env.filters["poster_variant"] = poster_variant
# Every page embeds the session's CSRF token (see _base.html), so forms and
# fetch() calls on any page can send it back.
templates.env.globals["csrf_token"] = session_csrf_token
templates.env.globals["google_enabled"] = google_enabled
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
async def index(
    request: Request,
    user: User | None = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    from .stripe_payments import supporter_since

    flash = request.session.pop("flash", None)
    flash_type = request.session.pop("flash_type", "success")
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "flash": flash,
            "flash_type": flash_type,
            "user": user,
            # Someone who has already given is thanked, not asked again.
            "supporter": supporter_since(db, user) is not None,
        },
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
    tilt: str | None = None,
    years: list[int] | None = None,
) -> dict[str, Any]:
    """What the results page needs to adjust a search without starting over.

    ``params`` are the active, non-default query parameters (never ``seed``:
    refining wants a fresh draw). Each chip names one active filter and links
    to the same search without it. ``shuffle_url`` repeats the search as is.
    ``suggestions`` are one-click nudges ("Older", "More obscure") relative to
    the films on screen; *years* are those films' release years.
    """
    from statistics import median
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
    if tilt in TILT_POPULARITY:
        params["tilt"] = str(tilt)
        labels["tilt"] = TILT_LABELS[str(tilt)]

    def url(changes: dict[str, str | None]) -> str:
        merged = {**params, **{k: v for k, v in changes.items() if v is not None}}
        for key, value in changes.items():
            if value is None:
                merged.pop(key, None)
        return "/recommend?" + urlencode(merged)

    suggestions: list[dict[str, str]] = []
    if years:
        middle = int(median(years))
        # "Older" means older than what is on screen, not older than some
        # fixed year: on a page of 1970s films it should reach further back.
        # A bound that would contradict the opposite filter drops that filter.
        older: dict[str, str | None] = {"max_year": str(middle - 1)}
        if min_year is not None and min_year > middle - 1:
            older["min_year"] = None
        newer: dict[str, str | None] = {"min_year": str(middle + 1)}
        if max_year is not None and max_year < middle + 1:
            newer["max_year"] = None
        suggestions.append(
            {"label": "Older", "url": url(older), "title": f"Films from before {middle}"}
        )
        suggestions.append(
            {"label": "Newer", "url": url(newer), "title": f"Films from after {middle}"}
        )
    for value, label in TILT_LABELS.items():
        if value != tilt:
            # The popularity signal only exists in the unified ranking.
            suggestions.append(
                {
                    "label": label,
                    "url": url({"tilt": value, "strategy": "unified"}),
                    "title": TILT_TITLES[value],
                }
            )

    chips = [
        {
            "key": key,
            "label": label,
            "remove_url": "/recommend?" + urlencode({k: v for k, v in params.items() if k != key}),
        }
        for key, label in labels.items()
    ]
    return {
        "clear_url": "/recommend?" + urlencode({"mood": mood, "strategy": strategy}),
        "params": {k: v for k, v in params.items() if k != "mood"},
        "chips": chips,
        "suggestions": suggestions,
        "shuffle_url": "/recommend?" + urlencode(params),
    }


DEFAULT_RESULT_LIMIT = 6


def _blank_to(default: Any) -> BeforeValidator:
    return BeforeValidator(lambda v: default if isinstance(v, str) and not v.strip() else v)


# A browser submits every field of a GET form, filled in or not, so the home
# page's search arrives as ``min_year=&max_year=&min_rating=``. Read literally,
# an empty string is not a number and the whole search was answered with a 422.
# A filter left blank means "no filter".
_BLANK_IS_UNSET = _blank_to(None)
_BLANK_IS_DEFAULT_LIMIT = _blank_to(DEFAULT_RESULT_LIMIT)
# The bounds sit on the inner type: placed on the optional one, they are
# applied to the None a blank field becomes.
_Year = Annotated[int, Field(ge=1900, le=2100)]
_Rating = Annotated[float, Field(ge=0, le=10)]

# One-click nudges to how much fame counts. The value replaces the unified
# ranking's popularity weight (0.20 by default). Measured on the corpus: the
# median IMDb vote count of results is about 136k at 0.0, 194k at the default
# and 271k at 0.45.
TILT_POPULARITY: dict[str, float] = {"obscure": 0.0, "mainstream": 0.45}
TILT_LABELS: dict[str, str] = {"mainstream": "More mainstream", "obscure": "More obscure"}
TILT_TITLES: dict[str, str] = {
    "mainstream": "Favour widely seen films",
    "obscure": "Stop favouring widely seen films",
}

# Valid strategy values accepted by the UI. Both search the local corpus; an
# unknown value (including the retired "keyword") falls back to the default.
STRATEGY_LABELS: dict[str, str] = {
    "semantic": "Semantic Search",
    "unified": "Unified (AI + Diversity)",
}


# Bumped by hand whenever the text of the privacy page changes.
PRIVACY_UPDATED = "20 November 2026"


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    """Browsers ask for this path whatever the page links to (and so do feed
    readers and bookmarks), so it is served here and not only under /static."""
    return FileResponse(
        STATIC_DIR / "favicon.ico",
        media_type="image/x-icon",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/privacy", response_class=HTMLResponse)
def privacy_page(request: Request, user: User | None = Depends(get_current_user)) -> HTMLResponse:
    """What the site stores and how to remove it. Google's OAuth consent
    screen requires a public URL for this before it can leave testing."""
    return templates.TemplateResponse(
        request, "privacy.html", {"user": user, "updated": PRIVACY_UPDATED}
    )


def _named_films(mood: str) -> list[str]:
    """Titles the query names ("like Hereditary"), for the note above the
    results. The search itself resolves them again; this is display only and
    must never break the page."""
    try:
        from .services.corpus import get_corpus_and_embeddings
        from .services.title_reference import find_title_references

        corpus, _ = get_corpus_and_embeddings()
        reference = find_title_references(mood, corpus)
        return list(reference.titles) if reference else []
    except Exception:  # noqa: BLE001
        logger.warning("Could not check the query for film titles", exc_info=True)
        return []


@app.get("/recommend", response_class=HTMLResponse)
async def ui_recommendations(
    request: Request,
    mood: str = Query(..., min_length=1),
    strategy: str | None = Query(default=None),
    min_year: Annotated[_Year | None, _BLANK_IS_UNSET, Query()] = None,
    max_year: Annotated[_Year | None, _BLANK_IS_UNSET, Query()] = None,
    min_rating: Annotated[_Rating | None, _BLANK_IS_UNSET, Query()] = None,
    limit: Annotated[int, _BLANK_IS_DEFAULT_LIMIT, Query(ge=1, le=20)] = DEFAULT_RESULT_LIMIT,
    english: Annotated[int | None, _BLANK_IS_UNSET, Query()] = None,
    seed: Annotated[
        int | None,
        _BLANK_IS_UNSET,
        Query(description="reproducible results; also disables sampling noise"),
    ] = None,
    tilt: str | None = Query(
        default=None, description="'mainstream' or 'obscure': how much popularity counts"
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
    tilt = tilt if tilt in TILT_POPULARITY else None
    if tilt:
        strategy_key = "unified"  # the only ranking with a popularity signal

    # A shared link names the films themselves. A seed cannot do this job:
    # the set on screen is a random draw, possibly personalised, and replaying
    # a seed gives the recipient a different one.
    shared = await run_in_threadpool(films_by_ids, ids.split(",")) if ids else []

    # A repeat of the same search should bring different films. Not applied
    # to a seeded request, which must stay reproducible.
    just_seen = recently_shown(request, mood) if seed is None and not shared else set()

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
        if tilt:
            weights = {**(weights or DEFAULT_WEIGHTS), "popularity": TILT_POPULARITY[tilt]}
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
            demote_ids=taste.demote_ids | just_seen,
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
            demote_ids=just_seen,
        )

    if movies and seed is None and not shared:
        remember_shown(request, mood, movies)

    # When ranking was personalised, say which liked film a result resembles.
    # Like every other part of personalisation, this must never break search.
    if movies and not shared and strategy_key == "unified":
        try:
            taste_for_page = _load_taste(db, user)
            if taste_for_page.taste_vector is not None:
                from .services.corpus import get_corpus_and_embeddings

                corpus_now, embeddings_now = get_corpus_and_embeddings()
                because = explain_by_liked_films(
                    movies, taste_for_page.liked_ids, corpus_now, embeddings_now
                )
                for movie in movies:
                    movie["because_liked"] = because.get(str(movie.get("imdb_id") or ""))
        except Exception:  # noqa: BLE001
            logger.warning("Could not explain results by liked films", exc_info=True)

    # Nothing came back and filters were on: work out which filter did it, so
    # the page can say so instead of blaming the mood.
    relief: dict[str, int] = {}
    if not movies and (min_year or max_year or min_rating is not None or english):
        try:
            relief = await run_in_threadpool(
                films_without_each_filter,
                min_year=min_year,
                max_year=max_year,
                min_rating=min_rating,
                english_only=bool(english),
            )
        except Exception:  # noqa: BLE001 - an explanation must never break the page
            relief = {}

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
            "named_films": [] if shared else _named_films(mood),
            "relief": relief,
            # A shared set was chosen by someone else, so what a viewer does
            # with it says nothing about the ranker; it is not measured.
            "result_key": (
                None
                if shared or looks_like_bot(request)
                else record_impression(db, request, mood, strategy_key, movies)
            ),
            "refine": refine_context(
                mood=mood,
                strategy=strategy_key,
                min_year=min_year,
                max_year=max_year,
                min_rating=min_rating,
                limit=limit,
                english=bool(english),
                tilt=tilt,
                years=[
                    int(str(m["year"])[:4])
                    for m in movies
                    if str(m.get("year") or "")[:4].isdigit()
                ],
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
            recommend_movies_advanced,
            mood=mood,
            limit=limit,
            seed=seed,
            # As documented: a seed makes the result reproducible. Without
            # this the seed only fixed the noise, it did not remove it.
            temperature=0.0 if seed is not None else 1.0,
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


@app.get("/api/providers/{imdb_id}")
async def api_providers(
    imdb_id: str,
    request: Request,
    region: str | None = Query(default=None, min_length=2, max_length=2),
) -> dict[str, Any]:
    """Where a film can be streamed, rented or bought, from the stored snapshot.

    The region is the one asked for, or a guess from the browser's language.
    Answers 404 when nothing is known, so the page simply shows no section.
    """
    from .services.providers import providers_for, region_from_accept_language

    chosen = (region or region_from_accept_language(request.headers.get("accept-language"))).upper()
    found = providers_for(imdb_id, chosen)
    if found is None:
        raise HTTPException(status_code=404, detail="No availability data for this film")
    return found


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


# How many recently shown films are remembered per mood, and for how many
# moods. Three pages' worth: a film sits out about three repeats of a search,
# then becomes eligible again. Small on purpose, because this lives in the
# session cookie.
RECENT_FILMS_PER_MOOD = 18
RECENT_MOODS = 3


def _mood_key(mood: str) -> str:
    import hashlib

    return hashlib.sha256(mood.strip().lower().encode()).hexdigest()[:16]


def recently_shown(request: Request, mood: str) -> set[str]:
    """Films this visitor was just shown for this same mood.

    Scoped to the mood on purpose. Searching again for the same thing should
    bring different films; searching for something else should not be denied
    its best match because it happened to appear a minute ago.
    """
    seen = request.session.get("seen")
    if not isinstance(seen, dict):
        return set()
    ids = seen.get(_mood_key(mood))
    return {str(i) for i in ids} if isinstance(ids, list) else set()


def remember_shown(request: Request, mood: str, movies: list[dict[str, Any]]) -> None:
    """Add the films just shown to the session's short per-mood memory."""
    seen = request.session.get("seen")
    if not isinstance(seen, dict):
        seen = {}
    key = _mood_key(mood)
    previous = seen.pop(key, [])
    previous = previous if isinstance(previous, list) else []
    shown = [str(m["imdb_id"]) for m in movies if m.get("imdb_id")]
    merged = [i for i in previous if i not in shown] + shown
    seen[key] = merged[-RECENT_FILMS_PER_MOOD:]
    # Dicts keep insertion order, so the oldest moods are first.
    while len(seen) > RECENT_MOODS:
        seen.pop(next(iter(seen)))
    request.session["seen"] = seen


def visitor_id_for(request: Request) -> str:
    """A random id for this browser session; not tied to an account or an IP."""
    import secrets as _secrets

    visitor_id = request.session.get("vid")
    if not visitor_id:
        visitor_id = _secrets.token_hex(16)
        request.session["vid"] = visitor_id
    return str(visitor_id)


def record_impression(
    db: Session, request: Request, mood: str, strategy: str, movies: list[dict[str, Any]]
) -> str | None:
    """Note which films were shown, and return the set's key for the page.

    Best-effort: this is measurement, and it must never cost a visitor their
    search results. Any failure is logged and the page renders without it.
    """
    imdb_ids = [str(m["imdb_id"]) for m in movies if m.get("imdb_id")]
    if not imdb_ids:
        return None
    key = set_result_key(mood, imdb_ids)
    try:
        visitor_id = visitor_id_for(request)
        seen = (
            db.query(Impression.id)
            .filter(Impression.visitor_id == visitor_id, Impression.result_key == key)
            .first()
        )
        if not seen:
            db.add(
                Impression(
                    visitor_id=visitor_id,
                    result_key=key,
                    mood=mood[:512],
                    strategy=strategy[:64],
                    imdb_ids=imdb_ids,
                )
            )
            db.commit()
    except Exception:  # noqa: BLE001
        db.rollback()
        logger.warning("Could not record the impression", exc_info=True)
    return key


def set_result_key(mood: str, imdb_ids: list[str]) -> str:
    """Identifies one result set for one mood, order included."""
    import hashlib

    return hashlib.sha256("\x00".join([mood.strip().lower(), *imdb_ids]).encode()).hexdigest()


@app.post("/api/set-feedback", response_model=SetFeedbackOut, dependencies=[Depends(require_csrf)])
async def submit_set_feedback(
    payload: SetFeedbackIn,
    request: Request,
    user: User | None = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Record "good picks" (+1) or not (-1) for a whole result set.

    Open to signed-out visitors: that is most of the audience, and the point
    is to collect any signal at all. Pressing the same verdict again removes
    it; pressing the other one switches it.

    The films are taken on trust from the page. Nothing here proves they were
    really shown for that mood, so the export treats this as weak evidence and
    keeps only ids that exist in the corpus.
    """
    visitor_id = visitor_id_for(request)
    key = set_result_key(payload.mood, payload.imdb_ids)
    existing = (
        db.query(SetFeedback)
        .filter(SetFeedback.visitor_id == visitor_id, SetFeedback.result_key == key)
        .first()
    )
    if existing:
        if existing.verdict == payload.verdict:
            db.delete(existing)
            db.commit()
            return {"status": "removed", "verdict": 0}
        existing.verdict = payload.verdict
        db.commit()
        return {"status": "updated", "verdict": payload.verdict}

    db.add(
        SetFeedback(
            visitor_id=visitor_id,
            result_key=key,
            user_id=user.id if user else None,
            mood=payload.mood[:512],
            strategy=payload.strategy[:64] if payload.strategy else None,
            imdb_ids=list(payload.imdb_ids),
            verdict=payload.verdict,
        )
    )
    db.commit()
    return {"status": "created", "verdict": payload.verdict}


@app.post("/api/interaction", response_model=InteractionOut, dependencies=[Depends(require_csrf)])
async def record_interaction(
    payload: InteractionIn, request: Request, db: Session = Depends(get_db)
) -> dict[str, str]:
    """Record that this visitor opened, clicked through to, or saved a film.

    Accepted only for a set this same visitor was actually shown, and only
    for a film in it. That keeps the table to things that happened on a real
    page, and lets the server fill in the rank itself instead of trusting it.
    """
    from sqlalchemy.exc import IntegrityError

    visitor_id = visitor_id_for(request)
    impression = (
        db.query(Impression)
        .filter(Impression.visitor_id == visitor_id, Impression.result_key == payload.result_key)
        .first()
    )
    if impression is None or payload.imdb_id not in impression.imdb_ids:
        return {"status": "ignored"}

    db.add(
        Interaction(
            visitor_id=visitor_id,
            result_key=payload.result_key,
            imdb_id=payload.imdb_id,
            kind=payload.kind,
            position=impression.imdb_ids.index(payload.imdb_id),
        )
    )
    try:
        db.commit()
    except IntegrityError:
        db.rollback()  # already recorded: the same action twice is one signal
        return {"status": "duplicate"}
    return {"status": "recorded"}


# Routers
app.include_router(auth_router)
app.include_router(email_verification_router)
app.include_router(google_router)
app.include_router(account_router)
app.include_router(onboarding_router)
app.include_router(password_reset_router)
app.include_router(history_router)
app.include_router(stripe_router)
app.include_router(watchlist_router)
app.include_router(public_list_router)
