#!/usr/bin/env python3
"""Turn real user feedback into an evaluation set.

The current benchmark is 15 hand-written moods whose expected answers are
famous films. That bias has already produced two misleading results:

* a 1,200-film corpus scored *worse* than 500, because obscure films compete
  for the same slots without being what the gold list names;
* moods score 0.000 while returning films that are genuinely correct -- the
  body-horror query surfaces *The Brood*, *Re-Animator* and *The Void*, none of
  which appear in its gold list.

Both are false negatives in the benchmark, not failures of the ranker, and
neither can be fixed by tuning against that same benchmark. Real like/dislike
feedback is ground truth: a liked film for a given mood *is* a correct answer.

Usage:
    python scripts/export_eval_set.py --min-likes 3 -o data/eval_set_from_feedback.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db import get_db_session  # noqa: E402
from app.models import MovieFeedback, SetFeedback  # noqa: E402


def set_verdict_titles(
    rows: list[Any],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Titles from set-level verdicts, keyed by mood: (thumbs-up, thumbs-down).

    Ids come from the page on trust, so only those present in the corpus are
    kept; an id that resolves to nothing is dropped, not guessed at.
    """
    from app.services.corpus import load_corpus

    title_of = {m.get("imdb_id"): (m.get("title") or "").strip().lower() for m in load_corpus()}
    up: dict[str, set[str]] = defaultdict(set)
    down: dict[str, set[str]] = defaultdict(set)
    for mood, imdb_ids, verdict in rows:
        if not mood:
            continue
        titles = {title_of[i] for i in (imdb_ids or []) if title_of.get(i)}
        (up if verdict > 0 else down)[mood.strip()].update(titles)
    return up, down


def build_cases(
    liked: dict[str, set[str]],
    disliked: dict[str, set[str]],
    set_liked: dict[str, set[str]],
    set_disliked: dict[str, set[str]],
    min_likes: int,
) -> list[dict[str, Any]]:
    """Graded eval cases from both kinds of feedback.

    Grade 2 is a film someone explicitly liked for that mood. Grade 1 is a
    film that was merely part of a set someone called good: real evidence,
    but about six films at once, so weaker. An explicit dislike outranks a
    set-level thumbs-up for the same film.
    """
    cases: list[dict[str, Any]] = []
    for mood in sorted(set(liked) | set(set_liked)):
        negative = disliked.get(mood, set())
        strong = liked.get(mood, set()) - negative
        weak = set_liked.get(mood, set()) - strong - negative
        gold = strong | weak
        if len(gold) < min_likes:
            continue
        cases.append(
            {
                "mood": mood,
                "gold": sorted(gold),
                "grades": {**{t: 1 for t in sorted(weak)}, **{t: 2 for t in sorted(strong)}},
                # Disliked films are recorded too: a ranker that surfaces them
                # is wrong in a way a gold list alone cannot express.
                "negative": sorted(negative),
                # Films from sets marked "not good": each one is only weakly
                # implicated, so they are kept apart from explicit dislikes.
                "negative_weak": sorted(set_disliked.get(mood, set()) - gold - negative),
                "source": "user_feedback",
            }
        )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Export an eval set from user feedback")
    parser.add_argument(
        "--min-likes",
        type=int,
        default=3,
        help="minimum liked films before a mood is worth scoring",
    )
    parser.add_argument("-o", "--output", default="data/eval_set_from_feedback.json")
    parser.add_argument(
        "--merge-gold",
        action="store_true",
        help="also fold in the hand-written gold set from the notebook",
    )
    args = parser.parse_args()

    liked: dict[str, set[str]] = defaultdict(set)
    disliked: dict[str, set[str]] = defaultdict(set)

    with get_db_session() as db:
        rows = db.query(MovieFeedback.mood, MovieFeedback.title, MovieFeedback.rating).all()
        set_rows = db.query(SetFeedback.mood, SetFeedback.imdb_ids, SetFeedback.verdict).all()

    set_liked, set_disliked = set_verdict_titles(set_rows)

    for mood, title, rating in rows:
        # Feedback recorded outside a search has no mood to attribute it to.
        if not mood or not title:
            continue
        (liked if rating > 0 else disliked)[mood.strip()].add(title.strip().lower())

    cases = build_cases(liked, disliked, set_liked, set_disliked, args.min_likes)
    skipped = len(set(liked) | set(set_liked)) - len(cases)
    if args.merge_gold:
        import ast

        from build_corpus import EVAL_NOTEBOOK

        tree = ast.parse(EVAL_NOTEBOOK.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "TEST_SET" for t in node.targets
            ):
                for case in ast.literal_eval(node.value):
                    cases.append({**case, "negative": [], "source": "handwritten"})
                break

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cases, indent=2, ensure_ascii=False))

    print(f"moods with feedback : {len(liked)}")
    print(f"skipped (<{args.min_likes} likes) : {skipped}")
    print(f"exported cases      : {len(cases)} -> {out}")
    if not cases:
        print(
            "\nNo mood has enough ratings yet. This becomes useful once the app\n"
            "has real usage; until then the handwritten set is all there is."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
