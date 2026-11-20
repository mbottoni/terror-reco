# Recommendation Engine

This document explains how TerrorReco matches a typed mood to horror films: the two strategies, the scoring signals, how variety is kept, and why results differ between searches.

## Design philosophy

- **Meaning, not keyword lists.** A sentence-transformer compares the query with each film's text, so "slow dread in the countryside" can find folk horror without a hand-written mapping.
- **Corpus only.** A pre-built collection of films is searched. Nothing on the request path calls TMDB or OMDb; the strategies that did were removed.
- **Built offline.** The corpus is assembled by `scripts/build_corpus.py`, never inside a request. See [Architecture](architecture.md).
- **Non-deterministic by design.** Controlled randomness means a repeat search shows different, still relevant films.
- **Measured.** Ranking changes are checked against an evaluation set, and the honest interval on its headline number is about ±0.06 NDCG, so small "gains" are treated as noise.

## Strategies

### 1. Semantic search

**Route:** `strategy=semantic`
**Function:** `recommend_movies_advanced()` in `app/services/recommender.py`

1. Apply the filters (year range, minimum rating, English only) to the corpus. Filtering first, not after retrieval, means a strict filter still gets the best matches among the films that pass.
2. Embed the query (cached per process) and take cosine similarity against the cached corpus embeddings.
3. Add temperature-scaled Gaussian noise to the scores of the top candidates.
4. Keep one film per franchise (see below).
5. The page is the top of that ranking.

### 2. Unified (AI + diversity), the default

**Route:** `strategy=unified`
**Function:** `recommend_unified_semantic()` in `app/services/unified_recommender.py`

Takes a 60-film pool from the semantic search and re-ranks it by a blend:

| Signal | Weight | Description |
|--------|--------|-------------|
| Semantic | 0.45 | Sentence-transformer cosine similarity |
| Keyword | 0.20 | Okapi BM25 over title, keywords, genre and overview. Document frequencies come from the whole corpus, not the pool |
| Popularity | 0.20 | IMDb rating × log(1 + votes), plus a small Metascore term |
| Recency | 0.05 | Weak prior favouring newer films |
| Taste | 0.15 | Signed-in users with at least three likes: cosine to the mean embedding of their liked films. Added on top, not renormalised |

Then:

1. Add Gaussian noise to the blended scores.
2. Keep one film per franchise, decided on the blended score.
3. Apply **Maximal Marginal Relevance** with a slightly jittered lambda to trade relevance against variety.
4. Films the user disliked are demoted below everything else, not removed, so a heavy rater still gets a full page.

Two alternatives were measured and deliberately left off: tuned blend weights and cross-encoder re-ranking. Both gains sat inside the noise; see `docs/evaluation-baseline.md`.

### Queries that name a film

"The Witch", "movies like Midsommar", "like Hereditary but less bleak", "Alien meets The Thing". When a query names a film in the corpus (`find_title_references()` in `app/services/title_reference.py`), both strategies change what they search with:

- The search vector is the named film's stored vector, blended 70/30 with the embedding of the rest of the query when there is any. Two named films are averaged.
- The unified strategy matches keywords against the named film's tags and keywords plus the rest of the query, never against the title's words.
- The named film and the rest of its franchise are left out, and the results page says so.

A title counts only where the query's shape says it is one: the whole query (optionally with a year), after a cue such as "like" or "similar to", or at the start before "but", "meets" or "vibes". A single-word title that is also an ordinary word ("Halloween", "Smile", "It") typed alone is read as a mood. Two films with the same title resolve to the stated year, else the better known. On the 24 title queries of the graded benchmark this moved NDCG@6 from 0.20 to 0.61; see `docs/evaluation-baseline.md`.

### What was removed

Two earlier strategies fetched from OMDb during the request: a TF-IDF ranker (`embedding`) and a title-keyword search (`keyword`). TF-IDF fitted a vectoriser on 30 to 120 documents per request, where IDF is meaningless, and both scored below the corpus pipeline. An old link with `strategy=keyword` now falls back to the default.

### One film per franchise

Sequels share keywords, genre and plot vocabulary, so they sit close together in embedding space. MMR alone does not separate them: a relevant sequel still wins a slot beside its sibling. Each corpus film records its TMDB collection, and both strategies keep only the best-ranked entry of a collection. Before this rule, 20% of semantic pages and 12% of unified pages repeated a franchise. TMDB groups remakes with their originals, so those count as one franchise too.

### Shared links

A link produced by "Share these results" carries the film ids (`ids=tt...,tt...`). The page then shows exactly those films and does no ranking. A seed cannot do this: the set on screen is a random, possibly personalised draw.

## ML Models

### Sentence-Transformers (Bi-Encoder)

- **Default model:** `sentence-transformers/all-mpnet-base-v2`
- **Embedding dimension:** 768
- **Used for:** Encoding the query and each film's composed text (`embedding_text()`: title, genre, keywords and overview) into the same vector space. Keywords matter most: plot summaries contain almost none of the tone and subgenre words people type, and adding them moved NDCG@6 from 0.31 to 0.50. The stored vector for a film is that document's vector with 30% of a second vector mixed in, the film's keywords encoded alone, so a short keyword list is not drowned out by a long overview (`mix_keyword_vectors()`; +0.04 NDCG@6 on the graded benchmark). That second vector also carries each film's descriptive tags (`data/film_tags.json`: 140 fixed tags such as "slow burn" or "very gory", written by an AI model), which supply the tone and subgenre words TMDB keywords lack; the unified strategy also matches query words against them. Worth a further +0.05 to +0.07.
- **Similarity metric:** Cosine similarity (embeddings are L2-normalised).
- **In the image:** the model is downloaded when the Docker image is built, and the container runs with the hub in offline mode. In development it is cached in `models/`.

The Marimo notebooks (`notebooks/2-embedding-models.py`) compare several alternatives:

| Model | Dim | Notes |
|-------|-----|-------|
| `all-mpnet-base-v2` | 768 | Best quality, default |
| `all-MiniLM-L6-v2` | 384 | Faster, slightly lower quality |
| `BAAI/bge-small-en-v1.5` | 384 | Competitive quality, small |
| `BAAI/bge-base-en-v1.5` | 768 | Alternative to mpnet |

### Cross-Encoders (evaluated but not in production)

The notebook `notebooks/3-cross-encoder.py` evaluates cross-encoder reranking:

- `cross-encoder/ms-marco-MiniLM-L-6-v2`
- `cross-encoder/ms-marco-TinyBERT-L-2-v2`

On the current 500-film corpus the cross-encoder gained +0.0026 NDCG, lost precision and added about 1.1 seconds per query, so it is off by default (`UNIFIED_USE_CROSS_ENCODER`). The notebook's larger gain was measured on an old 21-film corpus and should not be cited.

## The horror corpus

500 films, committed as `data/horror_corpus.json` and built offline by `scripts/build_corpus.py` (`make corpus`):

1. **Seed** the evaluation gold set, so the films the eval scores against are present.
2. **Discover** with TMDB (`/discover/movie?with_genres=27`), sorted by vote count, then a decade-stratified fill so the corpus is not all recent films. OMDb is not used for discovery: its search matches titles only and cannot browse a genre.
3. **Hydrate** each film from TMDB: plot, cast, crew, keywords, certification, franchise.
4. **Enrich** from OMDb with the IMDb rating and vote count, Metascore and awards. `vote_average` is always on the IMDb scale, which the rating filter and the popularity signal assume.
5. **Validate** size, duplicates, decade spread and gold coverage, then write the file.

Each record holds:

```
imdb_id, tmdb_id, title, overview, keywords, genre, poster_url, release_date, year,
vote_average (IMDb), imdbVotes, Metascore, rating_source, director, actors, writer,
runtime, language, country, rated, awards, collection_id, collection
```

A request never builds or extends the corpus. A 1,200-film version was tried and measured worse on the gold set, so the size stayed at 500; the gold set favours famous films, so that result says the larger corpus could not be shown to be better, not that it is worse.

## Non-Deterministic Recommendations

To avoid showing the same results every time for the same query, randomness is injected at two levels, and a repeat search for the same mood also demotes the films it just showed:

### 1. Score Perturbation (Semantic Search)

In `semantic_search()`, a `temperature` parameter controls Gaussian noise added to cosine similarity scores. A wider pool is fetched first, noise is applied proportional to the score spread, and the pool is re-ranked. This shuffles similar-scoring movies.

### 2. (Removed) Weighted random sampling

`recommend_movies_advanced()` used to draw the page at random from the top 18 films. Measured as served, that cost a third of the page's quality (NDCG@6 0.46 against 0.61) for variety the score noise already provides, and it also ran on the unified strategy's candidate pool. It was removed; see `docs/evaluation-baseline.md`.

### 3. Blended Noise + Jittered MMR (Unified Strategy)

In `recommend_unified_semantic()`:
- Gaussian noise is added to blended scores before ranking.
- The MMR diversity lambda is jittered by +/-0.08 each request, subtly changing the relevance-diversity trade-off.

## Filters

Users can apply these filters from the Advanced Filters panel:

| Filter | Parameter | Effect |
|--------|-----------|--------|
| Min year | `min_year` | Exclude movies before this year |
| Max year | `max_year` | Exclude movies after this year |
| Min IMDb rating | `min_rating` | Exclude movies below this IMDb score (0-10) |
| Results count | `limit` | Number of recommendations (1-20) |
| English only | `english` | Only show English-language titles |

Filters are applied to the corpus **before** ranking. Applied afterwards, a strict filter could leave only a few weak survivors out of the top results. When filters leave nothing at all, the results page names the filter responsible and how many films removing it would allow.

## Evaluation

- **Gold set:** 15 hand-written moods with expected films, in `notebooks/1-evaluation.py`.
- **`make eval`** (`scripts/run_eval.py`): Hit Rate, Precision, NDCG and MRR at 6, a paired comparison of the two strategies with a bootstrap confidence interval, and retrieval recall separately from ranking.
- **CI gate** (`tests/test_eval_gate.py`): a frozen slice of the eval, replayed on every push without the model. It fails if NDCG collapses or if the text that gets embedded changes.
- **From real use:** per-film likes, one-click verdicts on a whole result set, and anonymous usage data (what was shown, opened, clicked through to and saved) are exported as graded cases by `make eval-export`.

The limits are recorded in `docs/evaluation-baseline.md`: the gold set is small and favours famous films, and the honest interval on NDCG@6 is about ±0.06, so a ranking change smaller than that cannot be shown to help.

See [notebooks.md](notebooks.md) for the experiments.
