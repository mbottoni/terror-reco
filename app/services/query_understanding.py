"""What a query asks for, beyond its overall meaning.

One vector for the whole sentence cannot say that "campy fun with lots of
blood" wants two things at once, that "scary but with no gore" wants one thing
and excludes another, or that "80s slasher" states a fact about the release
year. Embedded whole, "not gory" lands next to "very gory".

This module reads a query into an :class:`Intent`:

* **wanted tags**, each with a weight: a phrase that is a tag's name or one of
  its plain-English aliases counts fully; a phrase the embedding model places
  close to a tag's name counts by that closeness;
* **avoided tags**: the same, for a phrase inside a negation ("no gore", "not
  a comedy", "sad more than scary"), with the opposite tag wanted where one
  exists ("no gore" wants "little to no gore");
* **a span of years**, from a decade ("the 80s", "1930s", "the eighties");
* **people**: a director, writer or cast member named in full, checked
  against the corpus's own credits;
* **extra words for keyword matching**: "Korean" also searches "Korea".

It is rule-based on purpose. Nothing here calls a model at request time
beyond the embedder the ranker already has, and every rule can be read.
It must not import from the rest of the app (see the Dockerfile).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# How close (cosine) a phrase must be to a tag's name to count as asking for it.
PHRASE_MATCH_FLOOR = 0.65
MAX_PHRASE_WORDS = 3

# Plain words people use for a tag. Deliberately short: each entry is a claim
# that the word means the tag, and a wrong one misleads every query using it.
ALIASES: dict[str, str] = {
    "gore": "very gory",
    "gory": "very gory",
    "bloody": "very gory",
    "blood": "very gory",
    "splattery": "very gory",
    "funny": "horror comedy",
    "comedy": "horror comedy",
    "hilarious": "horror comedy",
    "laughs": "horror comedy",
    "scary": "genuinely terrifying",
    "terrifying": "genuinely terrifying",
    "frightening": "genuinely terrifying",
    "sad": "melancholy",
    "depressing": "bleak",
    "atmosphere": "atmospheric",
    "space": "outer space",
    "twist": "twist ending",
    "twisty": "twist ending",
    "meta": "meta horror",
    "self-aware": "meta horror",
    "korean": "korean horror",
    "japanese": "j-horror",
    "italian": "italian horror",
    "spanish": "spanish-language horror",
    "silent": "silent era",
    "anthology": "anthology",
    "slow-burn": "slow burn",
    "slowburn": "slow burn",
    "zombie": "zombies",
    "vampire": "vampires",
    "werewolf": "werewolves",
    "ghost": "ghosts",
    "demon": "demons",
    "witch": "witches",
    "alien": "aliens",
    "slashers": "slasher",
}

# Excluding the left one asks for the right one.
OPPOSITES: dict[str, str] = {
    "very gory": "little to no gore",
    "splatter": "little to no gore",
    "extreme": "not too scary",
    "genuinely terrifying": "not too scary",
    "little to no gore": "very gory",
    "not too scary": "genuinely terrifying",
    "slow burn": "fast-paced",
    "fast-paced": "slow burn",
    "jump scares": "atmospheric",
}

# Adjectives of place also match the country a film comes from.
PLACES: dict[str, str] = {
    "korean": "korea",
    "japanese": "japan",
    "italian": "italy",
    "french": "france",
    "spanish": "spain",
    "german": "germany",
    "british": "united kingdom",
    "mexican": "mexico",
    "swedish": "sweden",
    "thai": "thailand",
    "indian": "india",
    "australian": "australia",
    "canadian": "canada",
}

_DECADE_WORDS = {
    "twenties": 1920,
    "thirties": 1930,
    "forties": 1940,
    "fifties": 1950,
    "sixties": 1960,
    "seventies": 1970,
    "eighties": 1980,
    "nineties": 1990,
}
_NEGATORS = {"no", "not", "without", "never", "non", "isnt", "arent", "less", "minus", "zero"}
# A negation covers the next few words and stops at one of these.
_BOUNDARIES = {"but", "and", "just", "only", "with", "that", "which", "or", "more"}
_NEGATION_REACH = 3
_STOPWORDS = frozenset("""
    a an the of to for in on at by with and or but that is are it its me my i
    something some movie movies film films horror about from more than not no
    without very really lots lot actually just kind like too so as be this
    """.split())


@dataclass(frozen=True)
class Intent:
    """What a query asks for besides its overall meaning."""

    want: dict[str, float] = field(default_factory=dict)  # tag -> weight in (0, 1]
    avoid: frozenset[str] = frozenset()
    years: tuple[int, int] | None = None
    extra_terms: tuple[str, ...] = ()  # appended to the keyword query
    people: tuple[str, ...] = ()  # directors, writers or cast named in the query

    def __bool__(self) -> bool:
        return bool(self.want or self.avoid or self.years or self.people)


def _tokens(query: str) -> list[str]:
    text = query.lower().replace("’", "'").replace("'", "")
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*|,", text)


def _years(tokens: list[str]) -> tuple[tuple[int, int] | None, set[int]]:
    """A decade named in the query, and the positions of its tokens."""
    for i, tok in enumerate(tokens):
        start: int | None = None
        match = re.fullmatch(r"(19|20)?(\d)0s", tok)
        if match:
            century, digit = match.group(1), int(match.group(2))
            if century:
                start = int(century) * 100 + digit * 10
            else:
                # "the 20s" today means the 2020s; "the 30s" to "the 90s" the last century.
                start = (2000 if digit <= 2 else 1900) + digit * 10
        elif tok in _DECADE_WORDS:
            start = _DECADE_WORDS[tok]
        if start is not None:
            return (start, start + 9), {i}
    return None, set()


def _negated_positions(tokens: list[str]) -> set[int]:
    """Positions of words inside a negation.

    "no gore", "not a comedy", "without jump scares" negate what follows.
    "sad more than scary" negates what follows "than": the query says which of
    two things it wants less of.
    """
    negated: set[int] = set()
    for i, tok in enumerate(tokens):
        opens = tok in _NEGATORS or (tok == "than" and i > 0 and tokens[i - 1] == "more")
        if not opens:
            continue
        reach = 0
        for j in range(i + 1, len(tokens)):
            word = tokens[j]
            if word == "," or word in _BOUNDARIES or word in _NEGATORS:
                break
            negated.add(j)
            if word not in _STOPWORDS:
                reach += 1
                if reach >= _NEGATION_REACH:
                    break
    return negated


def _phrases(tokens: list[str], skip: set[int]) -> list[tuple[str, tuple[int, ...]]]:
    """Runs of one to three words that could name a quality, with their positions."""
    out: list[tuple[str, tuple[int, ...]]] = []
    for size in range(1, MAX_PHRASE_WORDS + 1):
        for start in range(len(tokens) - size + 1):
            positions = tuple(range(start, start + size))
            words = [tokens[p] for p in positions]
            if "," in words or any(p in skip for p in positions):
                continue
            if words[0] in _STOPWORDS or words[-1] in _STOPWORDS:
                continue
            out.append((" ".join(words), positions))
    return out


PhraseMatcher = Callable[[list[str]], dict[str, tuple[str, float]]]


def person_key(name: str) -> str:
    """A credited name as it is compared: "Guillermo del Toro" -> "guillermo del toro"."""
    return " ".join(re.findall(r"[a-z0-9]+", name.lower().replace("'", "")))


def credited_people(film: dict[str, Any]) -> set[str]:
    """Everyone credited on *film* as director, writer or cast, as keys."""
    names: set[str] = set()
    for field_name in ("director", "writer", "actors"):
        for name in str(film.get(field_name) or "").split(","):
            key = person_key(name)
            if key:
                names.add(key)
    return names


def _people(tokens: list[str], known_people: frozenset[str]) -> tuple[tuple[str, ...], set[int]]:
    """Full names from the corpus's credits that appear in the query.

    Keyword matching alone cannot do this: it scores "vincent" and "price"
    separately, and "price" turns up in a dozen plots.
    """
    words = [tok for tok in tokens if tok != ","]
    positions = [i for i, tok in enumerate(tokens) if tok != ","]
    found: list[str] = []
    used: set[int] = set()
    for size in (4, 3, 2):
        for start in range(len(words) - size + 1):
            span = set(positions[start : start + size])
            if span & used:
                continue
            name = " ".join(w.replace("-", " ") for w in words[start : start + size])
            if name in known_people:
                found.append(name)
                used |= span
    return tuple(found), used


def understand(
    query: str,
    tag_names: list[str],
    match_phrases: PhraseMatcher | None = None,
    known_people: frozenset[str] = frozenset(),
) -> Intent:
    """Read *query* into an :class:`Intent`.

    *tag_names* is the vocabulary. *match_phrases*, when given, maps phrases to
    ``(closest tag, cosine)`` using the embedding model; without it only exact
    tag names and aliases are recognised, which is what tests use.
    """
    tokens = _tokens(query)
    if not tokens or len(tokens) > 40:
        return Intent()
    known = {name.lower(): name for name in tag_names}
    years, year_positions = _years(tokens)
    negated = _negated_positions(tokens)
    people, people_positions = _people(tokens, known_people) if known_people else ((), set())
    # A name is not also read for qualities: "Vincent Price" is not about price.
    phrases = _phrases(tokens, year_positions | people_positions)

    found: dict[str, tuple[float, bool]] = {}  # tag -> (weight, negated)

    def note(tag: str, weight: float, positions: tuple[int, ...]) -> None:
        is_negated = any(p in negated for p in positions)
        current = found.get(tag)
        if current is None or weight > current[0]:
            found[tag] = (weight, is_negated)

    unresolved: list[tuple[str, tuple[int, ...]]] = []
    for phrase, positions in phrases:
        spaced = phrase.replace("-", " ")
        if phrase in known or spaced in known:
            note(known.get(phrase) or known[spaced], 1.0, positions)
        elif phrase in ALIASES and ALIASES[phrase].lower() in known:
            note(known[ALIASES[phrase].lower()], 1.0, positions)
        else:
            unresolved.append((phrase, positions))

    if match_phrases is not None and unresolved:
        # A word that already resolved exactly is not also matched loosely as
        # part of a longer phrase: "very gory" should not also yield "splatter".
        settled = {
            p for phrase, positions in phrases for p in positions if _resolved(phrase, known)
        }
        candidates = [
            (phrase, positions)
            for phrase, positions in unresolved
            if not any(p in settled for p in positions)
        ]
        matches = match_phrases([phrase for phrase, _ in candidates]) if candidates else {}
        for phrase, positions in candidates:
            hit = matches.get(phrase)
            if hit and hit[1] >= PHRASE_MATCH_FLOOR and hit[0].lower() in known:
                note(known[hit[0].lower()], float(min(hit[1], 1.0)), positions)

    want = {tag: weight for tag, (weight, is_negated) in found.items() if not is_negated}
    avoid = {tag for tag, (_, is_negated) in found.items() if is_negated}
    for tag in list(avoid):
        opposite = OPPOSITES.get(tag)
        if opposite and opposite.lower() in known and opposite not in avoid:
            want.setdefault(known[opposite.lower()], 0.8)
    for tag in avoid:
        want.pop(tag, None)

    extra = tuple(PLACES[tok] for i, tok in enumerate(tokens) if tok in PLACES and i not in negated)
    return Intent(want=want, avoid=frozenset(avoid), years=years, extra_terms=extra, people=people)


def _resolved(phrase: str, known: dict[str, str]) -> bool:
    return (
        phrase in known
        or phrase.replace("-", " ") in known
        or (phrase in ALIASES and ALIASES[phrase].lower() in known)
    )


def film_year(film: dict[str, Any]) -> int | None:
    text = str(film.get("year") or film.get("release_date") or "")[:4]
    return int(text) if text.isdigit() else None


def aspect_scores(intent: Intent, films: list[dict[str, Any]]) -> tuple[list[float], list[bool]]:
    """For each film: how much of what the query wants it has (0 to 1), and
    whether it breaks something the query states (an avoided tag, the decade).
    """
    total = sum(intent.want.values()) + (1.0 if intent.years else 0.0) + len(intent.people)
    coverage: list[float] = []
    violates: list[bool] = []
    for film in films:
        tags = {t.strip() for t in str(film.get("tags") or "").split(",") if t.strip()}
        got = sum(weight for tag, weight in intent.want.items() if tag in tags)
        broken = bool(tags & intent.avoid)
        if intent.years:
            year = film_year(film)
            inside = year is not None and intent.years[0] <= year <= intent.years[1]
            got += 1.0 if inside else 0.0
            broken = broken or not inside
        if intent.people:
            credits = credited_people(film)
            named = sum(1 for person in intent.people if person in credits)
            got += float(named)
            # Naming someone states a fact about the film, like naming a decade.
            broken = broken or named == 0
        coverage.append(got / total if total else 0.0)
        violates.append(broken)
    return coverage, violates
