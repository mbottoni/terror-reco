# Recommendation Engine

This document explains how TerrorReco matches a typed mood to horror films: the two strategies, the scoring signals, how variety is kept, and why results differ between searches.

## Design philosophy

- **Meaning, not keyword lists.** A sentence-transformer compares the query with each film's text, so "slow dread in the countryside" can find folk horror without a hand-written mapping.
- **Corpus only.** A pre-built collection of films is searched. Nothing on the request path calls TMDB or OMDb; the strategies that did were removed.
- **Built offline.** The corpus is assembled by `scripts/build_corpus.py`, never inside a request. See [Architecture](architecture.md).
- **Non-deterministic by design.** Controlled randomness means a repeat search shows different, still relevant films.
- **Measured.** Ranking changes are checked against two benchmarks. The honest interval on a paired comparison is about ±0.034 NDCG@6 on the 55-mood graded benchmark and ±0.06 on the 15 hand-written moods, so smaller "gains" are treated as noise.

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
| Tags | 0.35 | How well the film's descriptive tags answer the query: cosine between the query and the best-matching of the film's tag definitions ("slasher: a killer stalks and picks off a group of victims..."). For a query that names a film, the share of tags in common with that film |
| Semantic | 0.30 | Sentence-transformer cosine similarity |
| Keyword | 0.25 | Okapi BM25 over title, keywords, tags, genre and overview. Document frequencies come from the whole corpus, not the pool |
| Popularity | 0.10 | IMDb rating × log(1 + votes), plus a small Metascore term |
| Recency | 0 | Fitted to zero; kept as a signal so a setting can turn it on |
| Taste | 0.15 | Signed-in users with at least three likes: cosine to the mean embedding of their liked films. Added on top, not renormalised |

The weights were fitted on the graded benchmark and checked on moods held out of the fit (+0.07 NDCG@6 over the previous hand-set blend). Re-fitting the old signals alone gained nothing; the tag signal is the gain. See `docs/evaluation-baseline.md`.

Then:

1. Add Gaussian noise to the blended scores.
2. Keep one film per franchise, decided on the blended score.
3. Apply **Maximal Marginal Relevance** with a slightly jittered lambda to trade relevance against variety.
4. Films the user disliked are demoted below everything else, not removed, so a heavy rater still gets a full page.

Two alternatives were measured and deliberately left off: cross-encoder re-ranking, and larger learned rankers (gradient boosting on 18 features did worse on held-out moods than the five-weight blend). See `docs/evaluation-baseline.md`.

### Reading the query

One vector for a whole sentence cannot say that "campy fun with lots of blood" wants two things, that "scary but with no gore" wants one and excludes another, or that "80s slasher" states a fact. Embedded whole, "not gory" lands next to "very gory". So before ranking, `app/services/query_understanding.py` reads the query into:

| Part | Example | How it is found |
|---|---|---|
| Wanted qualities | "campy", "lots of blood" → *campy*, *very gory* | A phrase that is a tag's name, one of a short list of plain-English aliases, or close to a tag's name in embedding space |
| Excluded qualities | "no gore", "not a comedy", "sad more than scary" | The same, inside a negation. The opposite tag is wanted where one exists ("no gore" wants *little to no gore*) |
| A decade | "the 80s", "1930s", "the eighties" | Pattern |
| People | "John Carpenter", "Vincent Price" | A full name found in the corpus's own director, writer and cast credits |
| Places | "Korean" also searches "Korea" | A short table |

The unified strategy then uses it three ways:

- **Candidates.** Besides the 60 nearest by meaning, the pool gets the best keyword matches (over a text that now includes director, writer, cast, country and language) and the films that satisfy everything the query asks for. A search by meaning cannot retrieve on a fact that is not in the embedded text.
- **The tag signal** becomes half "how much of what was asked does this film have" and half the best single tag.
- **Violations.** A film from the wrong decade, without the named person, or with an excluded quality has 0.35 taken off its score. It is moved down, not removed, so a page still fills when little fits.

It is rule-based on purpose: nothing calls a model at search time beyond the embedder already in use, and every rule can be read. What it does not understand it ignores, and the query is ranked as before.

### Queries that name a film

"The Witch", "movies like Midsommar", "like Hereditary but less bleak", "Alien meets The Thing". When a query names a film in the corpus (`find_title_references()` in `app/services/title_reference.py`), both strategies change what they search with:

- The search vector is the named film's stored vector, blended 70/30 with the embedding of the rest of the query when there is any. Two named films are averaged.
- The unified strategy matches keywords against the named film's tags and keywords plus the rest of the query, never against the title's words.
- The named film and the rest of its franchise are left out, and the results page says so.

A title counts only where the query's shape says it is one: the whole query (optionally with a year), after a cue such as "like" or "similar to", or at the start before "but", "meets" or "vibes". A single-word title that is also an ordinary word ("Halloween", "Smile", "It") typed alone is read as a mood. Two films with the same title resolve to the stated year, else the better known. On the 24 title queries of the graded benchmark this moved NDCG@6 from 0.20 to 0.64; see `docs/evaluation-baseline.md`.

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

On the 500-film corpus of the time the cross-encoder gained +0.0026 NDCG, lost precision and added about 1.1 seconds per query, so it is off by default (`UNIFIED_USE_CROSS_ENCODER`). It has not been re-measured since the tag signal and the larger corpus. The notebook's larger gain was measured on an old 21-film corpus and should not be cited.

## The horror corpus

650 films at the time of writing, committed as `data/horror_corpus.json`. The first 500 were built offline by `scripts/build_corpus.py` (`make corpus`):

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

Tags are not in the corpus file. `load_corpus()` attaches them from `data/film_tags.json` as a `tags` field and `save_corpus()` strips them again.

**It grows.** `scripts/grow_corpus.py` (`make grow`, and daily in GitHub Actions) adds the next horror films the corpus lacks, alternating TMDB's most-voted and best-rated lists, within OMDb's free daily quota. A new film gets predicted tags until a proper pass replaces them. See [corpus-growth.md](corpus-growth.md).

A request never builds or extends the corpus. An early 1,200-film version measured worse on the hand-written gold set, which can only reward famous films. Measured again on the graded benchmark, growing from 500 to 650 films changed page quality by nothing measurable (+0.009 NDCG@6, interval [-0.015, +0.035]) while widening what gets shown, so the corpus is now allowed to grow. New films have no grades, so benchmark numbers taken before and after a batch are not comparable until the new films are judged (`make eval-pool`).

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
| Tilt | `tilt` | `mainstream` raises the popularity weight to 0.45, `obscure` sets it to 0. Not a filter: it changes the blend, and forces the unified strategy |

Filters are applied to the corpus **before** ranking. Applied afterwards, a strict filter could leave only a few weak survivors out of the top results. When filters leave nothing at all, the results page names the filter responsible and how many films removing it would allow.

## Evaluation

- **Gold set:** 15 hand-written moods with expected films, in `notebooks/1-evaluation.py`.
- **`make eval`** (`scripts/run_eval.py`): Hit Rate, Precision, NDCG and MRR at 6, a paired comparison of the two strategies with a bootstrap confidence interval, and retrieval recall separately from ranking.
- **Graded benchmark** (`scripts/graded_eval.py`, `make eval-graded`): the queries in `eval/moods.json` scored against pooled judgments in `eval/judgments.json`, each film graded 0, 1 or 2 **by an AI judge, not a person**. Four groups of queries: 55 moods (the 15 hand-written plus 40 pooled), 24 that name a film (`graded_eval.py titles`), 23 that state constraints (`score --group constraint`) and 8 with no right answer (`nulls`). It has "as served" arms that go through the request path with the randomness on. A film a ranker returns that nobody graded is a **hole** and scores zero: run `make eval-pool`, grade what it lists, merge with `graded_eval.py add`, then score.
- **CI gate** (`tests/test_eval_gate.py`): a frozen slice of the eval, replayed on every push without the model. It fails if NDCG collapses or if the text that gets embedded changes. It replays without query understanding.
- **From real use:** per-film likes, one-click verdicts on a whole result set, and anonymous usage data (what was shown, opened, clicked through to and saved) are exported as graded cases by `make eval-export`.

The limits are recorded in `docs/evaluation-baseline.md`: the hand-written gold set is small and favours famous films; the graded benchmark has one AI judge, and the tags it rewards were also written by an AI model. A ranking change smaller than the paired interval (about ±0.034 on the graded moods) cannot be shown to help.

See [notebooks.md](notebooks.md) for the experiments.
