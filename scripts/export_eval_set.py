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
from app.models import MovieFeedback  # noqa: E402


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

    for mood, title, rating in rows:
        # Feedback recorded outside a search has no mood to attribute it to.
        if not mood or not title:
            continue
        (liked if rating > 0 else disliked)[mood.strip()].add(title.strip().lower())

    cases: list[dict[str, Any]] = []
    for mood, titles in sorted(liked.items()):
        if len(titles) < args.min_likes:
            continue
        cases.append(
            {
                "mood": mood,
                "gold": sorted(titles),
                # Disliked films are recorded too: a ranker that surfaces them
                # is wrong in a way a gold list alone cannot express.
                "negative": sorted(disliked.get(mood, set())),
                "source": "user_feedback",
            }
        )

    skipped = len(liked) - len(cases)
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
