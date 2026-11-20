from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from math import isfinite, log
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional in prod
    SentenceTransformer = None  # type: ignore[assignment,misc]

_SentenceTransformer: Any = SentenceTransformer  # keep as Any to avoid unreachable

_MODEL_CACHE: dict[str, Any] = {}

# Local model cache directory (avoids re-downloading from HuggingFace).
# Falls back to the default HF cache if the directory doesn't exist.
_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


def _normalize_text(s: str | None) -> str:
    return (s or "").strip().lower()


def _get_sbert(model_name: str = "sentence-transformers/all-mpnet-base-v2") -> Any:
    if _SentenceTransformer is None:
        return None
    if model_name not in _MODEL_CACHE:
        cache_folder = str(_MODELS_DIR) if _MODELS_DIR.is_dir() else None
        _MODEL_CACHE[model_name] = _SentenceTransformer(
            model_name,
            cache_folder=cache_folder,
        )
    return _MODEL_CACHE[model_name]


def _embed_sbert(texts: list[str]) -> np.ndarray:
    model = _get_sbert()
    if model is None:
        # Fallback: zeros; caller should handle low-signal gracefully
        return np.zeros((len(texts), 1), dtype=np.float32)
    vecs = model.encode(texts, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


_QUERY_CACHE_SIZE = 512
_QUERY_CACHE: OrderedDict[str, tuple[Any, np.ndarray]] = OrderedDict()


def embed_query(text: str) -> np.ndarray:
    """Embedding of one query string, shape ``(1, dim)``, cached per process.

    Queries repeat far more than they look like they should: the mood chips
    on the home page submit fixed strings, match chips and "more like this"
    re-run a keyword or a title, shared links replay a mood, and the unified
    strategy embeds the same mood twice in one request (once to retrieve,
    once to score). Each of those was a full model forward pass.

    An entry is only reused while ``_embed_sbert`` is the function that
    produced it, so swapping the embedder (tests, the eval gate) can never be
    answered from a stale vector. The array is read-only because it is shared
    between callers.
    """
    key = _normalize_text(text)
    embedder = _embed_sbert
    hit = _QUERY_CACHE.get(key)
    if hit is not None and hit[0] is embedder:
        _QUERY_CACHE.move_to_end(key)
        return hit[1]
    vec = embedder([key])
    vec.setflags(write=False)
    _QUERY_CACHE[key] = (embedder, vec)
    _QUERY_CACHE.move_to_end(key)
    while len(_QUERY_CACHE) > _QUERY_CACHE_SIZE:
        _QUERY_CACHE.popitem(last=False)
    return vec


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    result: np.ndarray = (a @ b.T).astype(np.float32)
    return result


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(np.min(x)), float(np.max(x))
    if not isfinite(lo) or not isfinite(hi) or abs(hi - lo) < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _popularity(detail: dict[str, Any]) -> float:
    rating = float(detail.get("vote_average") or 0.0)
    votes_str = (detail.get("imdbVotes") or detail.get("imdb_votes_raw") or "0").replace(",", "")
    metascore_str = detail.get("Metascore") or detail.get("metascore_raw") or "0"
    try:
        votes = int(votes_str)
    except Exception:
        votes = 0
    try:
        metascore = int(metascore_str)
    except Exception:
        metascore = 0
    return rating * (1 + log(1 + votes)) + 0.02 * metascore


# Common English words carry no retrieval signal but appear in nearly every
# plot, so the old overlap ratio was largely measuring how many stopwords a
# query contained.
_STOP_WORDS = frozenset(
    """a an the and or but of in on at to for with from by as is are was were be been being
    it its this that these those he she they them his her their who whom which what when
    where why how all any both each few more most other some such no nor not only own same
    so than too very can will just about into over after before between during above below
    up down out off again further then once""".split()
)


def _tokenize(text: str) -> list[str]:
    return [w for w in _normalize_text(text).split() if w not in _STOP_WORDS and len(w) > 2]


def _item_text(item: dict[str, Any]) -> str:
    """Text used for lexical matching -- includes keywords when present."""
    parts = [
        item.get("title") or "",
        item.get("tags") or "",
        item.get("keywords") or "",
        item.get("genre") or "",
        item.get("overview") or "",
    ]
    if not QUERY_UNDERSTANDING:
        return " ".join(p for p in parts if p)
    parts += [
        # Facts people search by: "John Carpenter movies", "with Vincent
        # Price", "Korean horror". None of them is in the embedded text.
        item.get("director") or "",
        item.get("writer") or "",
        item.get("actors") or "",
        item.get("country") or "",
        item.get("language") or "",
    ]
    return " ".join(p for p in parts if p)


@dataclass(frozen=True)
class Bm25Stats:
    """Collection statistics BM25 needs: how common each term is, overall."""

    n_docs: int
    avg_len: float
    df: dict[str, int]


def _bm25_stats(items: list[dict[str, Any]]) -> Bm25Stats:
    docs = [_tokenize(_item_text(it)) for it in items]
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set(doc))
    n = len(docs)
    return Bm25Stats(n_docs=n, avg_len=(sum(len(d) for d in docs) / n if n else 0.0), df=dict(df))


_CORPUS_STATS_CACHE: dict[str, Any] = {}


def corpus_bm25_stats() -> Bm25Stats | None:
    """BM25 statistics over the whole corpus, cached until the file changes.

    Returns None when there is no corpus, so callers fall back to the pool.
    Reads the corpus JSON directly rather than through
    ``get_corpus_and_embeddings()``: that would compute embeddings on a miss,
    and term counts need no model.
    """
    from .corpus import CORPUS_FILE, load_corpus

    try:
        mtime = CORPUS_FILE.stat().st_mtime if CORPUS_FILE.exists() else 0.0
        if _CORPUS_STATS_CACHE.get("mtime") != mtime:
            corpus = load_corpus()
            _CORPUS_STATS_CACHE.update(mtime=mtime, stats=_bm25_stats(corpus) if corpus else None)
        stats: Bm25Stats | None = _CORPUS_STATS_CACHE["stats"]
        return stats
    except Exception:  # noqa: BLE001 - never fail ranking over a statistics miss
        return None


def _bm25_scores(
    mood: str,
    items: list[dict[str, Any]],
    k1: float = 1.5,
    b: float = 0.75,
    stats: Bm25Stats | None = None,
) -> Any:
    """Okapi BM25 of *mood* against each item's text.

    Replaces a raw ``|query ∩ doc| / |query|`` overlap ratio, which had no
    stopword filtering, no term-frequency saturation and no inverse document
    frequency -- so a query's common words dominated the score.

    *stats* supplies the collection statistics (document frequencies and mean
    length). Production passes the whole corpus. Left as None they are taken
    from *items*, which is only right when *items* is itself the collection:
    over a 60-film candidate pool every film was retrieved **because** it
    resembles the query, so the query's own terms look common, their IDF
    collapses, and the same film scores differently depending on which query
    retrieved it.
    """
    query = _tokenize(mood)
    docs = [_tokenize(_item_text(it)) for it in items]
    scores = np.zeros(len(docs), dtype=np.float32)
    if not query or not docs:
        return scores

    if stats is None or stats.n_docs == 0:
        stats = _bm25_stats(items)
    n = stats.n_docs
    avg_len = stats.avg_len or 1.0
    doc_counters = [Counter(d) for d in docs]

    for term in query:
        n_q = stats.df.get(term, 0)
        # BM25 idf, floored at 0 so terms in most documents cannot go negative.
        # A term the collection has never seen (n_q == 0) is maximally rare;
        # it only scores if a candidate actually contains it.
        idf = max(0.0, log((n - n_q + 0.5) / (n_q + 0.5) + 1.0))
        if idf == 0.0:
            continue
        for i, counter in enumerate(doc_counters):
            freq = counter.get(term, 0)
            if not freq:
                continue
            norm = 1 - b + b * (len(docs[i]) / avg_len)
            scores[i] += idf * (freq * (k1 + 1)) / (freq + k1 * norm)
    return scores


def _lexical_sim_matrix(items: list[dict[str, Any]]) -> np.ndarray:
    """Pairwise Jaccard over title+overview tokens (fallback when no embeddings)."""
    token_sets = [
        set((_normalize_text(it.get("title")) + " " + _normalize_text(it.get("overview"))).split())
        for it in items
    ]
    n = len(items)
    matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = token_sets[i], token_sets[j]
            union = len(si | sj)
            val = (len(si & sj) / union) if union else 0.0
            matrix[i, j] = matrix[j, i] = val
    return matrix


def first_of_each_collection(items: list[dict[str, Any]]) -> list[int]:
    """Indices of *items* to keep so no franchise appears twice.

    *items* must already be best-first: the highest-ranked entry of each TMDB
    collection survives and its sequels are dropped. Films outside any
    collection are always kept.

    MMR cannot do this job. Sequels share keywords, genre and plot vocabulary,
    so they sit close in embedding space, but MMR only trades similarity
    against relevance; a strongly relevant sequel still wins a slot beside its
    sibling. On a six-film page, two entries from one franchise is a visibly
    poor set.
    """
    seen: set[Any] = set()
    keep: list[int] = []
    for index, item in enumerate(items):
        collection = item.get("collection_id")
        if collection is not None:
            if collection in seen:
                continue
            seen.add(collection)
        keep.append(index)
    return keep


def _mmr(
    items: list[dict[str, Any]],
    sims: np.ndarray,
    k: int,
    lambda_: float,
    embeddings: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Maximal Marginal Relevance: balance relevance against redundancy.

    Redundancy is measured on the sentence-transformer embeddings when they
    are available.  The previous implementation used Jaccard overlap of
    title+overview *word sets*, which measured vocabulary reuse rather than
    semantic similarity, and rebuilt those sets inside an O(k*n) loop.  The
    embedding path computes one pairwise matrix up front instead.
    """
    n = len(items)
    if n <= k:
        return items

    if embeddings is not None and embeddings.shape[0] == n:
        # Embeddings are L2-normalised, so the dot product is cosine similarity.
        sim_matrix = (embeddings @ embeddings.T).astype(np.float32)
    else:
        sim_matrix = _lexical_sim_matrix(items)

    selected: list[int] = [int(np.argmax(sims))]
    candidates = set(range(n)) - {selected[0]}

    while len(selected) < k and candidates:
        idx = np.fromiter(candidates, dtype=np.int64)
        sel = np.asarray(selected, dtype=np.int64)
        # Redundancy of each candidate = its closest already-selected item.
        max_sim = sim_matrix[np.ix_(idx, sel)].max(axis=1)
        scores = lambda_ * sims[idx] - (1.0 - lambda_) * max_sim
        best = int(idx[int(np.argmax(scores))])
        selected.append(best)
        candidates.remove(best)

    return [items[i] for i in selected]


CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_cross_encoder() -> Any:
    if CROSS_ENCODER_MODEL in _MODEL_CACHE:
        return _MODEL_CACHE[CROSS_ENCODER_MODEL]
    try:
        from sentence_transformers import CrossEncoder
    except Exception:  # pragma: no cover - optional dependency path
        return None
    cache_folder = str(_MODELS_DIR) if _MODELS_DIR.is_dir() else None
    _MODEL_CACHE[CROSS_ENCODER_MODEL] = CrossEncoder(CROSS_ENCODER_MODEL, cache_folder=cache_folder)
    return _MODEL_CACHE[CROSS_ENCODER_MODEL]


def _cross_encoder_scores(mood: str, items: list[dict[str, Any]]) -> np.ndarray | None:
    """Score (query, document) pairs jointly.  Returns None if unavailable."""
    model = _get_cross_encoder()
    if model is None:
        return None
    from .corpus import embedding_text

    try:
        pairs = [(mood, embedding_text(it)) for it in items]
        return np.asarray(model.predict(pairs), dtype=np.float32)
    except Exception:  # pragma: no cover - never fail a request over a rerank
        import logging

        logging.getLogger(__name__).warning(
            "Cross-encoder rerank failed; using blend", exc_info=True
        )
        return None


# Fitted on the graded benchmark (a non-negative ridge over these signals,
# rounded). Scored on moods held out of the fit, the blend with the tag signal
# beat the previous hand-set one (semantic .45, keyword .20, popularity .20,
# recency .05) by +0.07 NDCG@6; refitting the old four signals alone gained
# nothing. Recency fitted to zero. See docs/evaluation-baseline.md.
DEFAULT_WEIGHTS: dict[str, float] = {
    "semantic": 0.30,
    "keyword": 0.25,
    "tags": 0.35,
    "popularity": 0.10,
    "recency": 0.0,
}

_TAG_VECTOR_CACHE: dict[str, Any] = {}


def tag_definition_texts() -> tuple[list[str], list[str]]:
    """The tag vocabulary and the text embedded for each: "tag: what it means".

    The meaning is included because it is what a query resembles: "a killer
    stalks and picks off a group of victims" is closer to how people describe
    a slasher than the word is. Measured: +0.08 with it against +0.06 without.
    """
    from .corpus import load_tag_vocabulary

    vocabulary = load_tag_vocabulary()
    tags = list(vocabulary)
    return tags, [_normalize_text(f"{tag}: {vocabulary[tag]}") for tag in tags]


def _tag_vectors() -> tuple[dict[str, int], np.ndarray] | None:
    """Embeddings of the tag definitions, cached while the embedder is the same."""
    embedder = _embed_sbert
    if _TAG_VECTOR_CACHE.get("embedder") is not embedder:
        tags, texts = tag_definition_texts()
        if not tags:
            return None
        _TAG_VECTOR_CACHE.update(
            embedder=embedder,
            index={tag: i for i, tag in enumerate(tags)},
            vectors=np.asarray(embedder(texts), dtype=np.float32),
        )
    return _TAG_VECTOR_CACHE["index"], _TAG_VECTOR_CACHE["vectors"]


# Off: queries are read as one vector and nothing else, as they were before.
# Exists so the benchmark can measure what reading them adds.
QUERY_UNDERSTANDING = True
_INTENT_CACHE: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
_INTENT_CACHE_SIZE = 256
# Share of the tag signal taken from covering what the query asks for, the
# rest staying with the best single tag. Measured on the mood benchmark.
ASPECT_SHARE = 0.5
# Taken off the blended score (which spans 0 to 1) of a film that breaks
# something the query states: the wrong decade, or a quality it excludes.
VIOLATION_PENALTY = 0.35
# Films added to the candidate pool because they match the query's words or
# satisfy everything it asks for, whatever their overall similarity.
LEXICAL_EXTRA = 20
ASPECT_EXTRA = 30


def _tag_name_vectors() -> tuple[list[str], np.ndarray] | None:
    """Embeddings of the bare tag names, for matching a phrase to a tag."""
    embedder = _embed_sbert
    if _TAG_VECTOR_CACHE.get("names_embedder") is not embedder:
        tags, _ = tag_definition_texts()
        if not tags:
            return None
        _TAG_VECTOR_CACHE.update(
            names_embedder=embedder,
            names=tags,
            name_vectors=np.asarray(embedder([_normalize_text(t) for t in tags]), dtype=np.float32),
        )
    return _TAG_VECTOR_CACHE["names"], _TAG_VECTOR_CACHE["name_vectors"]


def _match_phrases(phrases: list[str]) -> dict[str, tuple[str, float]]:
    """Each phrase's closest tag name and the cosine between them."""
    loaded = _tag_name_vectors()
    if loaded is None or not phrases:
        return {}
    names, vectors = loaded
    encoded = np.asarray(_embed_sbert([_normalize_text(p) for p in phrases]), dtype=np.float32)
    if encoded.shape[-1] != vectors.shape[-1]:
        return {}
    sims = encoded @ vectors.T
    best = sims.argmax(axis=1)
    return {
        p: (names[int(j)], float(sims[i, int(j)]))
        for i, (p, j) in enumerate(zip(phrases, best, strict=True))
    }


_PEOPLE_CACHE: dict[str, Any] = {}


def _known_people() -> frozenset[str]:
    """Every full name credited in the corpus, cached until the corpus changes."""
    from .corpus import get_corpus_and_embeddings
    from .query_understanding import credited_people

    try:
        corpus, _ = get_corpus_and_embeddings()
    except Exception:  # noqa: BLE001
        return frozenset()
    if _PEOPLE_CACHE.get("corpus") is not corpus:
        names = {name for film in corpus for name in credited_people(film) if " " in name}
        _PEOPLE_CACHE.update(corpus=corpus, names=frozenset(names))
    people: frozenset[str] = _PEOPLE_CACHE["names"]
    return people


def query_intent(mood: str) -> Any:
    """What *mood* asks for (see ``query_understanding``), cached per process.

    Never raises: a query that cannot be read is treated as asking for nothing
    in particular, which is how every query was treated before.
    """
    from .query_understanding import Intent, understand

    if not QUERY_UNDERSTANDING:
        return Intent()
    key = _normalize_text(mood)
    embedder = _embed_sbert
    hit = _INTENT_CACHE.get(key)
    if hit is not None and hit[0] is embedder:
        _INTENT_CACHE.move_to_end(key)
        return hit[1]
    try:
        tags, _ = tag_definition_texts()
        intent = understand(mood, tags, _match_phrases, _known_people()) if tags else Intent()
    except Exception:  # noqa: BLE001
        intent = Intent()
    _INTENT_CACHE[key] = (embedder, intent)
    while len(_INTENT_CACHE) > _INTENT_CACHE_SIZE:
        _INTENT_CACHE.popitem(last=False)
    return intent


def widen_pool(
    mood: str,
    pool: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    corpus_embeddings: np.ndarray,
    query_vector: np.ndarray,
    exclude_ids: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Films to add to the re-ranker's candidates, beyond the nearest by meaning.

    The semantic search cannot retrieve on a fact that is not in the embedded
    text: no vector says "directed by John Carpenter" or "released in the
    1930s". Two kinds of film are added: the best keyword matches over the
    fields BM25 reads, and films that satisfy everything the query asks for.
    *corpus* and *corpus_embeddings* are the rows the search ran over (already
    filtered), so a year or rating filter is respected here too.
    """
    from .query_understanding import aspect_scores
    from .title_reference import find_title_references

    try:
        if not QUERY_UNDERSTANDING:
            return []
        if not corpus or find_title_references(mood, corpus) is not None:
            return []
        skip = {str(m.get("imdb_id") or "") for m in pool} | set(exclude_ids or ())
        sims = (
            np.asarray(query_vector, dtype=np.float32).reshape(1, -1) @ corpus_embeddings.T
        ).ravel()
        intent = query_intent(mood)
        chosen: list[int] = []

        keyword_text = " ".join([mood, *intent.extra_terms])
        lexical = np.asarray(
            _bm25_scores(keyword_text, corpus, stats=corpus_bm25_stats()), dtype=np.float32
        )
        for row in np.argsort(-lexical)[: LEXICAL_EXTRA * 3]:
            if len(chosen) >= LEXICAL_EXTRA or lexical[row] <= 0:
                break
            if str(corpus[int(row)].get("imdb_id") or "") not in skip:
                chosen.append(int(row))

        if intent:
            coverage, violates = aspect_scores(intent, corpus)
            complete = [i for i, c in enumerate(coverage) if c >= 0.999 and not violates[i]]
            complete.sort(key=lambda i: -float(sims[i]))
            for row in complete[: ASPECT_EXTRA * 2]:
                if sum(1 for r in chosen if r in complete) >= ASPECT_EXTRA:
                    break
                if row not in chosen and str(corpus[row].get("imdb_id") or "") not in skip:
                    chosen.append(row)

        extras = []
        for row in chosen:
            film = dict(corpus[row])
            film["_semantic_score"] = float(sims[row])
            film["_embedding_row"] = row
            extras.append(film)
        return extras
    except Exception:  # noqa: BLE001 - a wider pool is an improvement, never a requirement
        import logging

        logging.getLogger(__name__).warning("Could not widen the candidate pool", exc_info=True)
        return []


def _item_tags(item: dict[str, Any]) -> list[str]:
    return [t.strip() for t in str(item.get("tags") or "").split(",") if t.strip()]


def tag_scores(
    items: list[dict[str, Any]],
    query_vec: np.ndarray | None = None,
    like_tags: set[str] | None = None,
) -> np.ndarray:
    """How well each film's descriptive tags answer the query.

    For a mood: the cosine between the query and the best-matching of the
    film's tag definitions. One strong tag is the evidence; averaging over a
    film's tags punishes films that are well described.

    For a query that names films (*like_tags*, the named films' tags): the
    share of tags in common (Jaccard).

    Zeros when the film has no tags or the vocabulary cannot be embedded: the
    signal then drops out of the blend, it never fails the request.
    """
    scores = np.zeros(len(items), dtype=np.float32)
    if like_tags is not None:
        for i, item in enumerate(items):
            own = set(_item_tags(item))
            if own and like_tags:
                scores[i] = len(own & like_tags) / len(own | like_tags)
        return scores
    if query_vec is None:
        return scores
    try:
        loaded = _tag_vectors()
    except Exception:  # noqa: BLE001 - never fail ranking over an optional signal
        loaded = None
    if loaded is None:
        return scores
    index, vectors = loaded
    query = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    if vectors.shape[-1] != query.shape[-1]:
        return scores
    affinity = vectors @ query
    for i, item in enumerate(items):
        rows = [index[t] for t in _item_tags(item) if t in index]
        if rows:
            scores[i] = float(affinity[rows].max())
    return scores


def compute_signals(
    mood: str, items: list[dict[str, Any]], taste_vector: np.ndarray | None = None
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Compute each ranking signal, min-max normalised, plus item embeddings.

    Split out of :func:`recommend_unified_semantic` so weight tuning can
    reuse the exact production signals instead of reimplementing them --
    a tuner that drifts from production produces weights that do not apply.

    Returns ``(signals, item_embeddings)``.
    """
    # Reuse the cached corpus vectors when the caller preserved the row hints
    # from semantic_search.  Re-encoding 60 candidates cost ~2.4s per unified
    # request for vectors that were already sitting in corpus_embeddings.npy.
    #
    # The fallback path embeds the SAME composed document the corpus uses:
    # embedding `overview` alone here once silently reverted the semantic
    # signal to plot-only text, so unified scored worse than the plain
    # semantic search feeding it.
    from .corpus import embedding_text, embeddings_for, get_corpus_and_embeddings
    from .title_reference import find_title_references, keyword_query, reference_vector

    mood_vec = embed_query(mood)
    keyword_text = mood
    like_tags: set[str] | None = None
    plot_vecs = None
    try:
        corpus, corpus_embeddings = get_corpus_and_embeddings()
        if corpus_embeddings.size:
            plot_vecs = embeddings_for(items, corpus_embeddings)
            # A query that names a film is scored against that film's vector
            # and vocabulary, as the search that built the pool did.
            reference = find_title_references(mood, corpus)
            if reference is not None:
                mood_vec = reference_vector(reference, corpus_embeddings, embed_query)
                keyword_text = keyword_query(reference, corpus)
                like_tags = {t for row in reference.rows for t in _item_tags(corpus[row])}
    except Exception:  # noqa: BLE001 - never fail ranking over a cache miss
        plot_vecs = None
        mood_vec = embed_query(mood)
        keyword_text = mood
        like_tags = None

    # A query that names a film is matched on that film's tags instead.
    intent = query_intent(mood) if like_tags is None else None
    if intent is not None and intent.extra_terms:
        keyword_text = " ".join([keyword_text, *intent.extra_terms])

    if plot_vecs is None or plot_vecs.shape[-1] != mood_vec.shape[-1]:
        docs = [_normalize_text(embedding_text(m)) for m in items]
        plot_vecs = _embed_sbert(docs)

    tag_signal = _minmax(tag_scores(items, query_vec=mood_vec, like_tags=like_tags))
    violates = np.zeros(len(items), dtype=np.float32)
    if intent:
        from .query_understanding import aspect_scores

        coverage, broken = aspect_scores(intent, items)
        tag_signal = _minmax(
            ASPECT_SHARE * np.asarray(coverage, dtype=np.float32) + (1 - ASPECT_SHARE) * tag_signal
        )
        violates = np.asarray(broken, dtype=np.float32)

    signals: dict[str, np.ndarray] = {
        "semantic": _minmax(_cosine(mood_vec, plot_vecs).ravel()),
        "keyword": _minmax(_bm25_scores(keyword_text, items, stats=corpus_bm25_stats())),
        "tags": tag_signal,
        # Not a weighted signal: subtracted in recommend_unified_semantic().
        "violates": violates,
        "popularity": _minmax(np.array([_popularity(it) for it in items], dtype=np.float32)),
    }

    rec = np.zeros(len(items), dtype=np.float32)
    years: list[int | None] = []
    for it in items:
        y = it.get("year") or it.get("release_date") or ""
        try:
            y_int = int(str(y)[:4])
        except Exception:
            y_int = None
        years.append(y_int)
    valid = [y for y in years if isinstance(y, int)]
    if valid:
        y_arr = np.array([y if isinstance(y, int) else min(valid) for y in years], dtype=np.int32)
        rec = _minmax(y_arr.astype(np.float32))
    signals["recency"] = rec

    # Affinity to the user's taste vector (mean embedding of liked films).
    if taste_vector is not None and taste_vector.shape[-1] == plot_vecs.shape[-1]:
        signals["taste"] = _minmax(_cosine(taste_vector.reshape(1, -1), plot_vecs).ravel())
    else:
        signals["taste"] = np.zeros(len(items), dtype=np.float32)

    return signals, plot_vecs


def blend_signals(signals: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Weighted sum of normalised signals (missing signals contribute 0)."""
    n = len(next(iter(signals.values())))
    total = np.zeros(n, dtype=np.float32)
    for name, weight in weights.items():
        if weight and name in signals:
            total += weight * signals[name]
    return total.astype(np.float32)


def recommend_unified_semantic(
    *,
    mood: str,
    items: list[dict[str, Any]],
    limit: int = 6,
    diversity_lambda: float = 0.7,
    weights: dict[str, float] | None = None,
    seed: int | None = None,
    temperature: float = 1.0,
    taste_vector: np.ndarray | None = None,
    demote_ids: set[str] | None = None,
    use_cross_encoder: bool = False,
) -> list[dict[str, Any]]:
    if not items:
        return []

    signals, plot_vecs = compute_signals(mood, items, taste_vector=taste_vector)

    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    blended = blend_signals(signals, w)
    # A film that breaks something the query states (the decade, "no gore")
    # is moved down, not removed: when nothing else fits it still fills the page.
    blended = blended - VIOLATION_PENALTY * signals["violates"]

    # Films the user explicitly disliked are pushed below everything else
    # rather than removed, so a heavily-rated user still gets a full page.
    if demote_ids:
        penalty = np.array(
            [1.0 if (it.get("imdb_id") or "") in demote_ids else 0.0 for it in items],
            dtype=np.float32,
        )
        blended = blended - penalty

    # Add controlled noise to blended scores so results vary per request.
    # Noise scale is proportional to score spread, keeping top picks near
    # the top while shuffling the mid-band.  temperature=0 disables both
    # noise sources, making the call reproducible for evaluation.
    rng = np.random.default_rng(seed)
    if temperature > 0:
        spread = float(blended.max() - blended.min()) if len(blended) > 1 else 0.0
        noise = rng.normal(0, spread * 0.06 * temperature, size=len(blended)).astype(np.float32)
        blended_noisy = blended + noise
        # Slightly jitter diversity lambda so MMR selection also varies
        lambda_ = max(0.3, min(1.0, diversity_lambda + rng.uniform(-0.08, 0.08)))
    else:
        blended_noisy = blended
        lambda_ = diversity_lambda

    order = np.argsort(-blended_noisy)
    # One film per franchise, decided on the blended score: the unified
    # ranker, not the retriever, chooses which entry represents a series.
    ranked = [items[i] for i in order]
    order = order[first_of_each_collection(ranked)]
    pool_idx = order[: max(10, limit * 5)]
    pool = [items[i] for i in pool_idx]
    pool_scores = blended_noisy[pool_idx]
    pool_embs = plot_vecs[pool_idx]

    # Optional cross-encoder rerank of the shortlist.  A cross-encoder reads
    # the query and document *together* rather than comparing two independent
    # vectors, so it is more accurate but far slower -- hence shortlist-only
    # and off by default.
    if use_cross_encoder:
        ce_scores = _cross_encoder_scores(mood, pool)
        if ce_scores is not None:
            pool_scores = _minmax(ce_scores)

    selected = _mmr(pool, sims=pool_scores, k=limit, lambda_=lambda_, embeddings=pool_embs)
    # Internal hints (_embedding_row, _semantic_score) must not reach templates
    # or the stored search history.
    return [{k: v for k, v in m.items() if not k.startswith("_")} for m in selected]
