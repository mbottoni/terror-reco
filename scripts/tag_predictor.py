"""Predict descriptive tags for films nobody has tagged yet.

The 140-tag vocabulary in ``data/film_tags.json`` was applied to the original
corpus by an AI model that knows the films. A film added later by
``grow_corpus.py`` has no tags, and tags carry a third of the ranking weight,
so until a proper pass is made it gets predicted ones:

* the tags its nearest already-tagged neighbours share (weighted by cosine
  similarity, cubed), blended with
* how close the film's own text is to each tag's definition.

Measured by leaving each of the 500 tagged films out in turn and predicting
it from the rest: tags that at least 40% of the neighbour weight agrees on are
right 68% of the time; the top three by blended score, 63%. Predicting eight
tags per film, the number the AI pass wrote, is right 45% of the time, so
this keeps few. On the first 95 new films it was used on, 54% of its tags were
confirmed by a proper pass afterwards. Predicted tags are a stopgap and are listed under
``predicted`` in the tag file so they can be found and replaced.

Both the new film and its neighbours are compared on vectors built **without**
tags: the stored corpus vectors include each film's tags, which a new film
does not have.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NEIGHBOURS = 20
SIMILARITY_POWER = 3
AGREEMENT = 0.40  # share of neighbour weight a tag needs
NEIGHBOUR_SHARE = 0.7  # against the film's own closeness to the tag definition
MIN_TAGS, MAX_TAGS = 3, 6


def _zscore(rows: np.ndarray) -> np.ndarray:
    result: np.ndarray = (rows - rows.mean(axis=1, keepdims=True)) / (
        rows.std(axis=1, keepdims=True) + 1e-9
    )
    return result


def predict_tags(
    new_vecs: np.ndarray,
    reference_vecs: np.ndarray,
    reference_tags: np.ndarray,
    tag_vecs: np.ndarray,
    tags: list[str],
) -> list[list[str]]:
    """Tags for each row of *new_vecs*.

    *reference_tags* is a 0/1 matrix, one row per reference film and one
    column per entry of *tags*. All vectors are unit length.
    """
    if len(new_vecs) == 0 or len(reference_vecs) == 0:
        return [[] for _ in range(len(new_vecs))]
    sims = new_vecs @ reference_vecs.T
    agreement = np.zeros((len(new_vecs), len(tags)), dtype=np.float32)
    for i, row in enumerate(sims):
        nearest = np.argsort(-row)[:NEIGHBOURS]
        weight = np.clip(row[nearest], 0.0, None) ** SIMILARITY_POWER
        agreement[i] = (weight[:, None] * reference_tags[nearest]).sum(axis=0) / (
            weight.sum() + 1e-9
        )
    blended = NEIGHBOUR_SHARE * _zscore(agreement) + (1 - NEIGHBOUR_SHARE) * _zscore(
        new_vecs @ tag_vecs.T
    )
    out: list[list[str]] = []
    for i in range(len(new_vecs)):
        ranked = [int(j) for j in np.argsort(-blended[i])]
        agreed = [j for j in ranked if agreement[i, j] >= AGREEMENT]
        chosen = (
            agreed + [j for j in ranked if j not in agreed][: max(0, MIN_TAGS - len(agreed))]
        )[:MAX_TAGS]
        out.append([tags[j] for j in chosen])
    return out


def tagless_vectors(films: list[dict[str, Any]]) -> np.ndarray:
    """Each film's vector as it would be stored if it had no tags."""
    from app.services.corpus import embedding_text, mix_keyword_vectors
    from app.services.unified_recommender import _embed_sbert, _normalize_text

    if not films:
        return np.zeros((0, 0), dtype=np.float32)
    documents = _embed_sbert([_normalize_text(embedding_text(m)) for m in films])
    keyword_texts = [(m.get("keywords") or "").strip() for m in films]
    keywords = _embed_sbert([_normalize_text(t) if t else "none" for t in keyword_texts])
    return mix_keyword_vectors(documents, keywords, np.array([bool(t) for t in keyword_texts]))


def write_tag_file(path: Path, data: dict[str, Any]) -> None:
    """Write the tag file in its house style: one film per line, so a diff
    shows the films that changed and nothing else."""
    head = {k: v for k, v in data.items() if k not in ("tags", "predicted")}
    text = json.dumps(head, indent=1, ensure_ascii=False)
    lines = [
        f"  {json.dumps(imdb_id)}: {json.dumps(film_tags, ensure_ascii=False)}"
        for imdb_id, film_tags in data.get("tags", {}).items()
    ]
    text = (
        text[: text.rstrip().rfind("}")].rstrip() + ',\n "tags": {\n' + ",\n".join(lines) + "\n }"
    )
    if data.get("predicted"):
        text += ',\n "predicted": ' + json.dumps(sorted(data["predicted"]))
    path.write_text(text + "\n}\n")


def tag_new_films(imdb_ids: list[str]) -> int:
    """Predict and store tags for the corpus films in *imdb_ids*. Returns how many."""
    from app.services.corpus import TAGS_FILE, load_corpus
    from app.services.unified_recommender import _embed_sbert, tag_definition_texts

    data = json.loads(TAGS_FILE.read_text())
    predicted_before = set(data.get("predicted", []))
    by_id = {str(m["imdb_id"]): m for m in load_corpus()}
    targets = [by_id[i] for i in imdb_ids if i in by_id]
    # Only films tagged by the knowledgeable pass teach the predictor;
    # predictions must not feed on predictions.
    reference = [
        m
        for imdb_id, m in by_id.items()
        if data["tags"].get(imdb_id) and imdb_id not in predicted_before and imdb_id not in imdb_ids
    ]
    if not targets or not reference:
        return 0
    tags, texts = tag_definition_texts()
    column = {tag: j for j, tag in enumerate(tags)}
    reference_tags = np.zeros((len(reference), len(tags)), dtype=np.float32)
    for i, film in enumerate(reference):
        for tag in data["tags"][str(film["imdb_id"])]:
            if tag in column:
                reference_tags[i, column[tag]] = 1.0
    guesses = predict_tags(
        tagless_vectors(targets),
        tagless_vectors(reference),
        reference_tags,
        np.asarray(_embed_sbert(texts), dtype=np.float32),
        tags,
    )
    for film, guess in zip(targets, guesses, strict=True):
        if guess:
            data["tags"][str(film["imdb_id"])] = guess
            predicted_before.add(str(film["imdb_id"]))
    data["predicted"] = sorted(predicted_before)
    write_tag_file(TAGS_FILE, data)
    return sum(1 for g in guesses if g)
