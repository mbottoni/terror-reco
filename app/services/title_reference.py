"""Queries that name a film: "like Hereditary but less bleak", "The Witch".

Embedded as free text, such a query is dominated by the title's own words: the
named film ranks first and the rest merely share vocabulary with its title.
The corpus already holds a good vector for that film, so when a query names
one, the search vector is built from the film's vector (blended with whatever
else the query says) and the film itself is left out of the results.

Detection is deliberately conservative. A missed reference falls back to the
ordinary text search, which is no worse than before; a false one hijacks a
mood ("a quiet place to hide", "ghost story for a rainy night") and returns
films about the wrong thing. So a title only counts when the query's shape
says it is a title: the whole query, after a cue such as "like", or at the
start followed by "but"/"meets"/"vibes".

This module must not import from the rest of the app (see the Dockerfile).
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

# Master switch and blend, module-level so the benchmark can compare settings.
ENABLED = True
# Share of the search vector taken from the named film when the query says
# something else as well ("... but less bleak"). With nothing else, it is 1.
FILM_SHARE = 0.7

# Single-word titles that are also ordinary words or common horror subjects.
# Typed alone they are read as a mood, not as the film; after a cue ("like
# Smile") they still count. A title missing from this list errs toward being
# read as the film.
COMMON_WORD_TITLES = frozenset("""
    alien aliens blade cargo coma crawl creep devil doom dracula frankenstein
    freaks frenzy halloween house hush it legion life ma magic mama martin old
    orphan ouija pearl possession prey priest psycho rabid raw return ring rings
    run saw scream seconds shivers signs sisters smile society species split
    sunshine them trap triangle underwater us x krampus piranha anaconda
    antichrist poltergeist nosferatu godzilla jigsaw apostle sinister insidious
    malignant repulsion annihilation peninsula overlord barbarian nope carrie
    christine stoker tusk slither tremors gremlins critters constantine
    """.split())

# Titles that are function words. Even after a cue they count only when typed
# with a capital ("movies like It"), since "I like it gory" is not a reference.
FUNCTION_WORD_TITLES = frozenset("it us them x old ma run life return house".split())

# Off: the named film is still left out of the results, but the search uses
# the query's own text. Exists so the benchmark can tell the two effects apart.
USE_FILM_VECTOR = True
# Whether the keyword signal of the unified strategy matches candidates against
# the named film's tags and keywords, and not only the rest of the query.
KEYWORDS_FROM_FILM = True

# Words before a title that mark it as one.
_CUES_BEFORE = (
    "in the vein of",
    "in the style of",
    "the same vibe as",
    "same vibe as",
    "the vibe of",
    "vibes of",
    "reminds me of",
    "reminiscent of",
    "inspired by",
    "similar to",
    "fans of",
    "fan of",
    "after watching",
    "just watched",
    "i loved",
    "i liked",
    "i enjoyed",
    "loved",
    "enjoyed",
    "such as",
    "crossed with",
    "mixed with",
    "meets",
    "a la",
    "like",
)
# Words after a title at the start of the query that mark it as one.
_CUES_AFTER = (
    "but",
    "meets",
    "vibes",
    "vibe",
    "style",
    "esque",
    "like",
    "type",
    "crossed",
    "mixed",
)

# Left over once the title and cue are removed; not worth embedding.
_FILLER = frozenset("""
    a an the and or of to for in on with me something anything movie movies film
    films horror like similar more another one ones want watch recommend find
    give show please i im looking some that this it its is are was really just
    kind sort stuff other others such as same vibe vibes style type but what
    next now
    """.split())

_MAX_TITLE_TOKENS = 9


def normalise(text: str) -> str:
    """Lowercase ASCII words, single-spaced: "Mother!" -> "mother"."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    text = text.replace("&", " and ").replace("'", "").replace("’", "")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


@dataclass(frozen=True)
class TitleReference:
    """Films a query names, and what the query says besides."""

    rows: tuple[int, ...]  # rows of the corpus the query was resolved against
    imdb_ids: tuple[str, ...]
    titles: tuple[str, ...]  # "Hereditary (2018)", for the results page
    exclude_ids: frozenset[str]  # the named films and the rest of their franchises
    residual: str  # the query without titles, cues and filler; may be empty
    query: str = ""  # the query as typed


def _votes(movie: dict[str, Any]) -> int:
    try:
        return int(str(movie.get("imdbVotes") or "0").replace(",", ""))
    except ValueError:
        return 0


def _title_index(corpus: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Normalised title -> rows. A leading "the" is optional for long titles."""
    index: dict[str, list[int]] = {}
    for row, movie in enumerate(corpus):
        title = normalise(str(movie.get("title") or ""))
        if not title:
            continue
        index.setdefault(title, []).append(row)
        tokens = title.split()
        if tokens[0] == "the" and len(tokens) >= 4:
            index.setdefault(" ".join(tokens[1:]), []).append(row)
    return index


def _pick(rows: list[int], corpus: list[dict[str, Any]], year: int | None) -> int:
    """Among films sharing a title: the stated year, else the best known."""
    if year is not None:
        for row in rows:
            if str(corpus[row].get("year") or "")[:4] == str(year):
                return row
    return max(rows, key=lambda r: _votes(corpus[r]))


def _match_at(
    tokens: list[str], start: int, index: dict[str, list[int]], *, fuzzy: bool
) -> tuple[int, list[int]] | None:
    """Longest title starting at ``tokens[start]``: (token count, rows)."""
    limit = min(_MAX_TITLE_TOKENS, len(tokens) - start)
    for size in range(limit, 0, -1):
        rows = index.get(" ".join(tokens[start : start + size]))
        if rows:
            return size, rows
    if fuzzy:
        # A misspelling ("hereditery", "midsomar"). Long words only, and only
        # where the query's shape already says a title is expected.
        for size in range(min(limit, 5), 0, -1):
            phrase = " ".join(tokens[start : start + size])
            if len(phrase) < 7:
                continue
            close = difflib.get_close_matches(phrase, list(index), n=1, cutoff=0.88)
            if close:
                return size, index[close[0]]
    return None


def _year_after(tokens: list[str], position: int) -> int | None:
    if position < len(tokens) and re.fullmatch(r"(19|20)\d\d", tokens[position]):
        return int(tokens[position])
    return None


def find_title_references(query: str, corpus: list[dict[str, Any]]) -> TitleReference | None:
    """The films *query* names, or None when it reads as an ordinary mood."""
    if not ENABLED or not corpus:
        return None
    tokens = normalise(query).split()
    if not tokens or len(tokens) > 40:
        return None
    index = _title_index(corpus)

    found: list[int] = []
    used = [False] * len(tokens)

    def take(start: int, size: int, rows: list[int]) -> None:
        year = _year_after(tokens, start + size)
        found.append(_pick(rows, corpus, year))
        for i in range(start, start + size + (1 if year else 0)):
            used[i] = True

    # 1. The whole query is a title, give or take a year and "movie".
    core = list(tokens)
    while core and core[-1] in ("movie", "film"):
        core.pop()
    year = _year_after(core, len(core) - 1) if len(core) > 1 else None
    if year is not None:
        core = core[:-1]
    whole = _match_at(core, 0, index, fuzzy=True) if core else None
    if whole and whole[0] == len(core):
        single_common = len(core) == 1 and core[0] in COMMON_WORD_TITLES and year is None
        if not single_common:
            row = _pick(whole[1], corpus, year)
            return _reference([row], corpus, "", query)

    # 2. A title right after a cue: "something like the witch but funnier".
    for cue in _CUES_BEFORE:
        cue_tokens = cue.split()
        n = len(cue_tokens)
        for start in range(len(tokens) - n):
            if tokens[start : start + n] != cue_tokens or any(used[start : start + n]):
                continue
            at = start + n
            # "like the movie hereditary"
            while at < len(tokens) - 1 and tokens[at] in ("the", "movie", "film", "films"):
                if _match_at(tokens, at, index, fuzzy=False):
                    break
                at += 1
            hit = _match_at(tokens, at, index, fuzzy=True)
            if hit and hit[0] == 1 and tokens[at] in FUNCTION_WORD_TITLES:
                typed = tokens[at].capitalize()
                if not re.search(rf"(?<![A-Za-z]){typed}(?![A-Za-z])", query):
                    hit = None
            if hit and not any(used[at : at + hit[0]]):
                for i in range(start, start + n):
                    used[i] = True
                take(at, hit[0], hit[1])

    # 3. A title opening the query, followed by a cue: "alien but underwater".
    if not used[0]:
        hit = _match_at(tokens, 0, index, fuzzy=False)
        if hit and hit[0] < len(tokens):
            after = hit[0] + (1 if _year_after(tokens, hit[0]) else 0)
            if after < len(tokens) and tokens[after] in _CUES_AFTER:
                take(0, hit[0], hit[1])
                used[after] = True
                # "alien meets the thing": the second title needs no cue of its own.
                if tokens[after] in ("meets", "crossed", "mixed"):
                    nxt = after + 1
                    if nxt < len(tokens) and tokens[nxt] == "with":
                        used[nxt] = True
                        nxt += 1
                    second = (
                        _match_at(tokens, nxt, index, fuzzy=False) if nxt < len(tokens) else None
                    )
                    if second and not any(used[nxt : nxt + second[0]]):
                        take(nxt, second[0], second[1])

    if not found:
        return None
    residual = " ".join(
        tok for tok, taken in zip(tokens, used, strict=True) if not taken and tok not in _FILLER
    )
    return _reference(list(dict.fromkeys(found)), corpus, residual, query)


_SEQUEL_MARK = re.compile(r"^(\d+|i{1,3}|iv|v|vi{0,3}|part|chapter)$")


def _same_series(named: str, other: str) -> bool:
    """A remake with the same title, or "Alien: Covenant" / "Halloween II" for
    "Alien" / "Halloween". TMDB's collections miss these: prequels and reboots
    are often filed as franchises of their own."""
    named, other = named.removeprefix("the "), other.removeprefix("the ")
    if other == named:
        return True
    if not other.startswith(named + " "):
        return False
    rest = other[len(named) + 1 :].split()
    # A numbered entry always; a subtitle only for titles distinctive enough
    # that the prefix is not a coincidence ("It" must not exclude "It Follows").
    return bool(_SEQUEL_MARK.match(rest[0])) or (len(named) >= 5 and named not in ("house",))


def _reference(
    rows: list[int], corpus: list[dict[str, Any]], residual: str, query: str
) -> TitleReference:
    films = [corpus[r] for r in rows]
    ids = tuple(str(f.get("imdb_id") or "") for f in films)
    collections = {f.get("collection_id") for f in films if f.get("collection_id") is not None}
    exclude = set(ids)
    named = [normalise(str(f.get("title") or "")) for f in films]
    for movie in corpus:
        title = normalise(str(movie.get("title") or ""))
        if movie.get("collection_id") in collections or any(_same_series(n, title) for n in named):
            exclude.add(str(movie.get("imdb_id") or ""))
    titles = tuple(
        f"{f.get('title')} ({str(f.get('year'))[:4]})" if f.get("year") else str(f.get("title"))
        for f in films
    )
    return TitleReference(
        rows=tuple(rows),
        imdb_ids=ids,
        titles=titles,
        exclude_ids=frozenset(exclude),
        residual=residual,
        query=query,
    )


def reference_vector(
    reference: TitleReference,
    corpus_embeddings: np.ndarray,
    embed: Callable[[str], np.ndarray],
) -> np.ndarray:
    """Search vector for a query that names films, shape ``(1, dim)``, unit length.

    The mean of the named films' vectors, blended with the embedding of the
    rest of the query when there is any.
    """
    if not USE_FILM_VECTOR:
        return embed(reference.query)
    film = corpus_embeddings[list(reference.rows)].mean(axis=0)
    film = film / (np.linalg.norm(film) or 1.0)
    if reference.residual:
        rest = embed(reference.residual).reshape(-1)
        rest = rest / (np.linalg.norm(rest) or 1.0)
        film = FILM_SHARE * film + (1.0 - FILM_SHARE) * rest
        film = film / (np.linalg.norm(film) or 1.0)
    out: np.ndarray = film.astype(np.float32).reshape(1, -1)
    return out


def keyword_query(reference: TitleReference, corpus: list[dict[str, Any]]) -> str:
    """Text for keyword matching: what the query says besides the titles, plus
    the named films' own tags and keywords. Never the title words, which only
    match the film itself and its sequels."""
    if not USE_FILM_VECTOR:
        return reference.query
    parts = [reference.residual]
    if KEYWORDS_FROM_FILM:
        for row in reference.rows:
            film = corpus[row]
            parts += [str(film.get("tags") or ""), str(film.get("keywords") or "")]
    return " ".join(p for p in parts if p)
