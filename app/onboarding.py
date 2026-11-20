"""Taste bootstrap: pick a few films you love, right after signing up.

Personalisation needs three likes before it does anything, and a like used to
require opening the detail view of a film a search had already returned. So
the feature only switched on for people who had stuck around long enough not
to need it. Here a new user ticks posters once and the taste vector exists
from their first search.

Each pick is an ordinary ``MovieFeedback`` row with ``rating=+1``: no new
table and no second ranking path. The rows carry no mood, so the eval export,
which only uses feedback tied to a search, ignores them.
"""

from __future__ import annotations

import logging
from math import log
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response

from .auth import _set_flash, get_current_user
from .db import get_db
from .models import MovieFeedback, User
from .security import require_csrf
from .services.personalization import MIN_LIKES_FOR_TASTE

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/onboarding", dependencies=[Depends(require_csrf)])

FILMS_PER_SUBGENRE = 3
MAX_PICKS = 60

# A spread across the genre, so one person's picks can describe a taste. The
# top thirty films by votes would be thirty films everybody has seen and
# say almost nothing. Each entry is (label, keyword fragments to look for).
SUBGENRES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Slasher", ("slasher",)),
    ("Folk horror", ("folk horror", "pagan", "cult")),
    ("Body horror", ("body horror", "mutation")),
    ("Found footage", ("found footage", "mockumentary")),
    ("Zombies", ("zombie",)),
    ("Vampires", ("vampire",)),
    ("Hauntings", ("haunted house", "ghost", "haunting")),
    ("Possession", ("possession", "exorcism", "demon")),
    ("Cosmic horror", ("lovecraft", "cosmic")),
    ("Sci-fi horror", ("alien", "space", "spaceship")),
    ("Psychological", ("psychological", "paranoia", "madness")),
    ("Creature features", ("monster", "creature", "werewolf")),
)


def _popularity(movie: dict[str, Any]) -> float:
    try:
        votes = int(str(movie.get("imdbVotes") or "0").replace(",", ""))
    except ValueError:
        votes = 0
    return float(movie.get("vote_average") or 0.0) * (1 + log(1 + votes))


def onboarding_films(
    corpus: list[dict[str, Any]], per_subgenre: int = FILMS_PER_SUBGENRE
) -> list[dict[str, Any]]:
    """The best-known films of each subgenre, as ``{"label", "films"}`` groups.

    Recognisable beats obscure here: a poster grid only works if people know
    what they are looking at. A film appears once even if it fits several
    subgenres, and a franchise is offered once, so the grid is not half sequels.
    """
    ranked = sorted(corpus, key=_popularity, reverse=True)
    used_films: set[str] = set()
    used_collections: set[Any] = set()
    groups: list[dict[str, Any]] = []
    for label, fragments in SUBGENRES:
        films: list[dict[str, Any]] = []
        for movie in ranked:
            if len(films) >= per_subgenre:
                break
            imdb_id = movie.get("imdb_id")
            collection = movie.get("collection_id")
            keywords = (movie.get("keywords") or "").lower()
            if (
                not imdb_id
                or not movie.get("poster_url")
                or imdb_id in used_films
                or (collection is not None and collection in used_collections)
                or not any(fragment in keywords for fragment in fragments)
            ):
                continue
            films.append(movie)
            used_films.add(imdb_id)
            if collection is not None:
                used_collections.add(collection)
        if films:
            groups.append({"label": label, "films": films})
    return groups


def _corpus() -> list[dict[str, Any]]:
    from .services.corpus import get_corpus_and_embeddings

    return get_corpus_and_embeddings()[0]


@router.get("/", response_class=HTMLResponse, response_model=None)
async def onboarding_page(
    request: Request,
    user: User | None = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    if not user:
        return RedirectResponse(url="/auth/login", status_code=303)
    groups = await run_in_threadpool(onboarding_films, await run_in_threadpool(_corpus))
    liked = {
        row.imdb_id
        for row in db.query(MovieFeedback.imdb_id).filter(
            MovieFeedback.user_id == user.id, MovieFeedback.rating > 0
        )
    }
    page: Response = request.app.state.templates.TemplateResponse(
        request,
        "onboarding.html",
        {
            "user": user,
            "groups": groups,
            "liked": liked,
            "min_likes": MIN_LIKES_FOR_TASTE,
            "flash": request.session.pop("flash", None),
            "flash_type": request.session.pop("flash_type", "success"),
        },
    )
    return page


@router.post("/")
async def save_picks(
    request: Request,
    user: User | None = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> RedirectResponse:
    if not user:
        return RedirectResponse(url="/auth/login", status_code=303)

    form = await request.form()
    picked = [value for value in form.getlist("imdb_id") if isinstance(value, str)][:MAX_PICKS]
    # Only films that are really in the corpus: the taste vector is built from
    # corpus embeddings, so anything else would be a like that does nothing.
    by_id = {m.get("imdb_id"): m for m in await run_in_threadpool(_corpus)}
    existing = {
        row.imdb_id: row for row in db.query(MovieFeedback).filter(MovieFeedback.user_id == user.id)
    }

    added = 0
    for imdb_id in dict.fromkeys(picked):
        movie = by_id.get(imdb_id)
        if movie is None:
            continue
        row = existing.get(imdb_id)
        if row is None:
            db.add(
                MovieFeedback(
                    user_id=user.id,
                    imdb_id=imdb_id,
                    title=str(movie.get("title") or "")[:512],
                    rating=1,
                    strategy="onboarding",
                )
            )
            added += 1
        elif row.rating < 0:
            # Picking a film here is a clearer statement than an old dislike.
            row.rating = 1
            added += 1
    db.commit()

    total = (
        db.query(MovieFeedback)
        .filter(MovieFeedback.user_id == user.id, MovieFeedback.rating > 0)
        .count()
    )
    logger.info("ONBOARDING: user_id=%s added=%d liked_total=%d", user.id, added, total)
    if total >= MIN_LIKES_FOR_TASTE:
        _set_flash(request, "Thanks. Your recommendations are now tuned to your taste.")
    elif total:
        more = MIN_LIKES_FOR_TASTE - total
        _set_flash(request, f"Saved. Pick {more} more any time to personalise your results.")
    return RedirectResponse(url="/", status_code=303)
