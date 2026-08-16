# Work Plan

Ordered by value per unit of effort. Measurements referenced here live in
`docs/evaluation-baseline.md`; latency figures were taken on this machine with a
warm model and the 500-film corpus.

Status: `[ ]` todo · `[x]` done · `[~]` in progress · `[-]` deliberately not doing

---

## P0 — Performance bug

- [x] **1. Stop re-embedding items that are already cached.** DONE — **2,475 ms -> 49 ms**,
  NDCG unchanged. Also added a process-level corpus/embedding cache.
  `compute_signals()` calls `_embed_sbert()` on the mood plus all 60 candidate
  items. Those 60 came *from the corpus*, whose embeddings are already cached in
  `corpus_embeddings.npy` — so every unified request pays a full mpnet forward
  pass to recompute vectors it already has.

  | strategy | measured |
  |---|---:|
  | semantic | 21 ms |
  | unified | **2,475 ms** (2,361 ms of it in `_embed_sbert`, 61 docs) |

  Thread the corpus row indices through `semantic_search()` so unified can slice
  the cached matrix. Expected ~2,475 ms → ~25 ms with identical output.
  Knock-on: the cross-encoder's "+46% latency" verdict was measured against an
  inflated baseline and should be re-tested afterwards.

## P1 — Correctness, security, and dead weight

- [x] **2. Bind CSRF tokens to the session.** Signature-only validation meant a token
  minted in an attacker's own session verified in a victim's. Now compared against
  the session copy. Regression test confirmed to fail against the old code.
- [x] **3. Add a health endpoint.** `/healthz` reports corpus + embedding readiness.
- [x] **4. Point `/api/recommendations` at the corpus path.** It defaulted to the
  keyword strategy, so the JSON API returned worse results than the UI.
- [x] **5. Retire the TF-IDF `embedding` strategy.** Meaningless IDF at 30-120 docs,
  live OMDb on the request path, worse than semantic. Also removed the only
  reason the undeclared PyYAML dependency existed.
- [x] **6. Remove `print()` session leakage.** Now `logger.info`.

## P2 — Structural

- [x] **7. Normalize `SearchHistory.results_json`.** New `search_results` table storing
  film *references* in rank order. The migration backfills existing blobs (verified
  against seeded legacy rows) so no history is lost.
- [x] **8. Build the Docker image from `uv.lock`.** Verified: image now ships
  starlette 0.52.1 / fastapi 0.128.4 (the tested versions) and torch 2.10.0**+cpu**
  with CUDA `None`, so the CPU-only trick survives. `make requirements` regenerates.
- [x] **9. Replace `datetime.utcnow`.** Now `_utcnow()` returning aware UTC, with
  `DateTime(timezone=True)` columns.

## Frontend

- [x] **10. Drop `/loading` for corpus strategies.** Its hardcoded 600 ms setTimeout
  was 92% of the wait for a 49 ms request. Only `keyword` still routes through it.
- [x] **11. "More like this".** `/api/similar/{imdb_id}` + a row in the detail modal.
  Pure item-to-item cosine on cached vectors -- no query encoding at all.
- [x] **12. Explain *why* a film matched.** Keyword chips on each card showing the
  terms the query actually hit. Plural folding is display-only -- applying it to
  `_tokenize` would change BM25 and invalidate the recorded eval numbers.
- [x] **13. Keyword chips as filters.** The match chips are now links that re-run
  the term as a search.
- [x] **14. No-JS fallback.** `<noscript>` IMDb link per card plus a banner; without
  JS the cards were dead divs with no way to reach the film.

## New features

- [x] **15. Shareable result permalinks.** "Share these results" pins a seed into
  the URL so the link reproduces that exact set instead of a fresh draw.
- [x] **16. Watchlist.** `watchlist` table, `/watchlist/` page, toggle API, Save
  button in the modal. Kept separate from feedback on purpose: a like is a taste
  signal about a film seen, a save is only an intent to watch, and mixing them
  would poison the taste vector with unwatched films.
- [-] **17. Grow the corpus.** Tried at 1,200 films and **measured worse** on every
  metric (ndcg 0.4997 -> 0.3437): the extra films are obscure and compete for the
  same six slots. Reverted to 500. The gold set is canon-biased, so this is not
  proof that a bigger corpus is worse for real users -- it is proof we cannot
  currently show it is better, which is the argument for item 19. Discovery work
  for 1,200 films is preserved in the checkpoint.
- [~] **18. Fix the two broken moods.** Root-caused, not yet fixed: the gold films
  for both (*Hereditary*, *The Fly*, *The Thing*, *Videodrome*) are present with
  keywords, so this is a ranking problem in an already-enriched corpus rather than
  missing vocabulary. Needs its own investigation.
  ORIGINAL: *body horror and grotesque
  transformation* and *slow-burn psychological dread* both score 0.000 with their
  gold films present in the corpus.
- [ ] **19. Harvest a real eval set from feedback.** 15 synthetic moods is what
  starves the current tuning; real ratings are ground truth.

## Deliberately not doing

- [-] **pgvector.** At 500 films the numpy dot product is microseconds. The
  benefit actually attributed to it — prefilter-then-rank — was delivered
  directly. Revisit past ~10k films.
- [-] **Tuned blend weights / cross-encoder as defaults.** Both measured; gains
  sit inside the run-to-run noise floor and the cross-encoder also costs
  precision. Wired to settings, off by default. See `docs/evaluation-baseline.md`.
