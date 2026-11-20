# Evaluation Baseline

Two measurements, both **2026-08-16**, both against the 500-film TMDB corpus.

## Result summary

| Metric | v1 baseline | v2 (composed embeddings) | Change |
|--------|------------:|-------------------------:|-------:|
| ndcg@6 (semantic, deterministic) | 0.3111 | **0.4997** | **+61%** |
| ndcg@6 (unified, n=10) | 0.3724 | **0.4896** | **+31%** |
| hit_rate@6 (semantic) | 0.6667 | **0.8667** | +0.200 |
| precision@6 (semantic) | 0.2778 | **0.4667** | +0.189 |
| moods scoring 0.000 | 5 of 15 | **2 of 15** | -3 |

**What changed in v2:** the embedded document went from `overview` alone to
`title + genre + keywords + overview`, where `keywords` are TMDB tags carrying the
tone/subgenre vocabulary ("isolation", "paranoia", "transformation") that plot summaries
lack. Also: BM25 replaced the stopword-dominated overlap ratio, and MMR diversity moved
from Jaccard-on-word-sets to cosine on embeddings.

The moods that were broken are the ones that moved:

| Mood | v1 | v2 |
|------|---:|---:|
| cosmic Lovecraftian isolation | 0.000 | 0.645 |
| eerie folk horror pagan rituals | 0.000 | 0.321 |
| home invasion and paranoia | 0.000 | 0.247 |
| campy fun with lots of blood | 0.000 | 0.108 |
| slasher with a masked killer | 0.454 | 0.870 |
| zombie apocalypse survival | 0.454 | 0.892 |

Not everything improved. `slow-burn psychological dread` regressed 0.151 -> 0.000 and
`body horror and grotesque transformation` stayed at 0.000, so two moods remain unsolved.

### A bug this measurement caught

`recommend_unified_semantic` re-embedded items from `overview` alone, so the unified
strategy silently reverted to the *old* weak semantic signal while the retrieval step
used the enriched one. Symptom: unified scored **worse** than the plain semantic search
feeding it (0.4103 vs 0.4997), and forcing `semantic=1.0` scored 0.3059 -- almost exactly
the v1 baseline. Fixed by embedding the same composed document; unified now leads again.

Ablation after the fix (deterministic, seed=0):

| Configuration | ndcg@6 |
|---|---:|
| semantic only | 0.4997 |
| unified, lambda=0.7 (default) | 0.5070 |
| unified, lambda=0.85 | 0.5365 |
| unified, lambda=1.0 (MMR off) | 0.5419 |

Turning MMR off scores highest, but diversity is the entire point of the "AI + Diversity"
strategy and the metric gives no credit for it, so the default stays at 0.7. Do not tune
lambda upward to chase this number.

## Corpus size: 1,200 films measured WORSE than 500

Growing the corpus from 500 to 1,200 films (decade-stratified fill) regressed every
metric:

| Metric | 500 films | 1,200 films | Change |
|--------|----------:|------------:|-------:|
| ndcg@6 (semantic) | 0.4997 | 0.3437 | **-0.156** |
| ndcg@6 (unified) | 0.4896 | 0.3983 | -0.091 |
| hit_rate@6 | 0.8667 | 0.7333 | -0.133 |
| precision@6 | 0.4667 | 0.3444 | -0.122 |

The 500-film corpus is canon-heavy by construction (`vote_count.desc` plus gold
seeding). The extra 700 films are more obscure, and they compete for the same six
slots: they can match a mood semantically while not being the answer the gold set
expects. **Reverted to 500 films.**

**Important caveat, and the reason this is not a verdict on corpus size:** the gold
set is 15 synthetic moods whose expected answers are famous films, so it is
structurally biased toward a small canon-heavy corpus. A larger corpus may well
serve real users better while scoring worse here. What this measurement actually
shows is that *we currently have no way to demonstrate a larger corpus helps* --
which is the argument for harvesting a real evaluation set from user feedback.

The 1,200-film discovery work is preserved in `data/.corpus_build_state.json`, so
this is cheap to revisit once a less biased benchmark exists.

## A corpus regression that shipped

Between commits `5b77ae6` and `8fabc85`, the corpus lost all 500 keyword fields and
NDCG would have fallen back to roughly the pre-keyword baseline. Nothing failed: no
test, no validation gate, no CI step.

Cause: `build_corpus.py` wrote `list(state["records"].values())` straight to disk.
Those checkpoint records were hydrated *before* the keyword field existed, and
`--refresh-keywords` had updated only the corpus file, not the checkpoint. Any later
`--resume` run therefore resurrected stale records and silently deleted the
enrichment behind the +61% gain.

Fixed in two places: the final write now merges with the corpus already on disk so
it can only add fields, never remove them; and `--refresh-keywords` writes back into
the checkpoint so the two cannot drift apart again.

## Two things measured and deliberately NOT shipped

Both were on the roadmap as expected wins. Both were measured and rejected.

### Tuned blend weights: rejected

`scripts/tune_weights.py` (`make tune`) grid-searches 77 weight combinations at
lambda=0.7, reusing the production `compute_signals()` so the result actually applies.

| ndcg@6 | semantic | keyword | popularity | recency |
|-------:|---------:|--------:|-----------:|--------:|
| 0.5197 | 0.45 | 0.20 | 0.25 | 0.10 |
| 0.5185 | 0.40 | 0.30 | 0.20 | 0.10 |
| 0.5183 | 0.45 | 0.30 | 0.25 | 0.00 |
| **0.5070** | **0.45** | **0.20** | **0.20** | **0.05** | <- current default |

Best config beats the default by **+0.0127**. The pipeline's own run-to-run stddev is
**~0.02**, and the top 12 configurations sit inside a 0.007 band. The "improvement" is
therefore inside the noise floor, and it was selected on all 15 moods with no holdout,
so it is an overfit to the test set rather than a real gain.

**Decision: defaults unchanged.** Weights are configurable via `UNIFIED_WEIGHTS` for
experimentation, but shipping the tuned values would be reporting noise as progress.

### Cross-encoder reranking: rejected as a default

| Configuration | ndcg@6 | precision@6 | time (15 moods) |
|---|---:|---:|---:|
| blend only | 0.5070 | 0.4889 | 36.7s |
| + cross-encoder rerank | 0.5096 | 0.4778 | 53.7s |

+0.0026 NDCG (inside noise), **worse** precision, and ~46% more latency -- roughly 1.1s
extra per query. `notebooks/3-cross-encoder.py` found a gain, but that was measured on
the old 21-film corpus with plot-only embeddings, where retrieval was weak enough to
leave headroom. With composed embeddings there is nothing left for it to fix.

**Decision: implemented and wired to `UNIFIED_USE_CROSS_ENCODER`, default off.**

## Retrieval or ranking? (2026-08-22)

`scripts/run_eval.py` now separates the two stages, because every metric above is
measured at k=6 *after* both of them, so a 0.000 could be either a retriever that
never found the film or a ranker that buried it.

Two numbers per mood, deterministic (`temperature=0`):

- **recall@60** -- of the gold films that are in the corpus, how many the 60-candidate
  pool actually contains.
- **ceiling** -- the best NDCG@6 obtainable from that pool by *any* re-ranker.

| Mood | ndcg@6 | ceiling | recall@60 |
|------|-------:|--------:|----------:|
| slow-burn psychological dread | **0.000** | 0.775 | 0.44 |
| body horror and grotesque transformation | **0.000** | 0.892 | 0.56 |
| campy fun with lots of blood | 0.108 | 0.494 | 0.29 |
| creepy kids and childhood fears | 0.117 | 0.645 | 0.38 |
| home invasion and paranoia | 0.247 | 0.892 | 0.71 |
| eerie folk horror pagan rituals | 0.321 | 0.892 | 0.62 |
| *(nine moods at ceiling 1.000)* | | 1.000 | 0.75--1.00 |
| **mean** | 0.4997 | | **0.737** |

**The two 0.000 moods are ranking failures, not retrieval failures.** Both have
ceilings well above zero: the pool holds 4 of 9 gold films for slow-burn dread and 5 of
9 for body horror, and the top-6 contains none of them. No amount of corpus work or
retrieval fusion can be what fixes those two -- the answers were already retrieved and
then ordered below six other films.

Two honest qualifications:

1. Section A ranks by pure cosine, so "ranking failure" here means the same similarity
   score that pulled the film into the top 60 also put six other films above it.
2. Part of that is the benchmark, not the ranker. Body horror returns *The Brood*,
   *Re-Animator* and *The Void* -- all canonically correct, none in the gold list. The
   films "burying" the gold ones are partly right answers the gold set does not know
   about.

Separately, **32 gold films are in the corpus but never reach the pool** (mean recall
0.737). That is a real retrieval loss, and it is invisible in every number recorded
above it on this page.

## Paired comparison: unified vs semantic is not established

The A/B table reports two independent means with a stddev over runs. That throws away
the fact that both arms saw the *same* 15 moods, and moods differ in difficulty far
more than runs differ in noise. Pairing per mood and bootstrapping over moods (5,000
resamples, 95% CI):

| Metric | mean delta (B - A) | 95% CI | verdict |
|--------|-------------------:|:------:|---------|
| hit_rate@6 | -0.0267 | [-0.2067, +0.1333] | indistinguishable |
| precision@6 | -0.0111 | [-0.0589, +0.0333] | indistinguishable |
| ndcg@6 | -0.0174 | [-0.0771, +0.0375] | indistinguishable |
| mrr | -0.0081 | [-0.1352, +0.1169] | indistinguishable |

**The unified pipeline is not measurably better or worse than plain semantic search on
this benchmark.** The earlier ±0.02 stddev understated the uncertainty because it only
counted run-to-run noise; once mood sampling is counted, the NDCG interval is ±0.06.

This does not mean unified should be removed -- MMR diversity is deliberate and this
metric gives it no credit (see the lambda ablation above). It means **this benchmark
cannot rank the two strategies against each other**, and any future change smaller than
about 0.06 NDCG cannot be evaluated with 15 synthetic moods however many times it is
re-run. That is an argument for widening the benchmark, not for re-running it.

---

## v1 baseline (pre-composed-embeddings)

Recorded **2026-08-16**, against the 500-film TMDB corpus.

Reproduce with:

```bash
make eval          # or: python scripts/run_eval.py --runs 10
```

`scripts/run_eval.py` imports the metric functions from `notebooks/1-evaluation.py`
via marimo's `Cell.run()`, so these numbers cannot drift from what the notebook reports.

## Headline numbers

15 moods, ~8 gold titles each, k=6.

| Metric | A: semantic (deterministic) | B: unified pipeline (n=10) |
|--------|----------------------------:|---------------------------:|
| hit_rate@6 | 0.6667 | 0.7067 ± 0.0442 |
| precision@6 | 0.2778 | 0.3589 ± 0.0186 |
| ndcg@6 | 0.3111 | 0.3724 ± 0.0159 |
| mrr | 0.5056 | 0.5150 ± 0.0358 |

**A** is `semantic_search(temperature=0)` taking the top 6 by cosine — fully reproducible,
so it is the number to track across changes.

**B** is the real `unified` pipeline. It is reported as mean ± stddev over 10 runs because
the pipeline injects noise at three independent points (score perturbation, weighted
sampling, MMR lambda jitter); a single run is not a measurement. Note the stddev is large
relative to the differences you would be trying to detect — **any future comparison needs
repeated runs, not one**.

Blending and MMR do earn their place: unified beats pure semantic on every metric,
most clearly on precision (+0.08) and NDCG (+0.06).

## The real finding: 5 of 15 moods score zero

| | |
|---|---|
| **Gold coverage ceiling** | **124/127 titles are in the corpus** |

Every mood scoring 0.000 NDCG has 7–9 of its gold films sitting in the corpus. These are
**ranking failures, not data gaps** — the films are there and the retriever does not
surface them.

| Mood | NDCG@6 | Gold in corpus |
|------|-------:|---------------:|
| haunted house with dark secrets | 0.892 | 9/9 |
| demonic possession and exorcism | 0.870 | 8/8 |
| survival horror isolated in nature | 0.645 | 9/9 |
| slasher with a masked killer | 0.454 | 8/8 |
| zombie apocalypse survival | 0.454 | 8/8 |
| sci-fi horror in space | 0.420 | 8/8 |
| found footage realistic terror | 0.342 | 9/9 |
| creepy kids and childhood fears | 0.247 | 8/8 |
| vampire gothic romance | 0.191 | 8/8 |
| slow-burn psychological dread | 0.151 | 9/9 |
| **campy fun with lots of blood** | **0.000** | 7/9 |
| **cosmic Lovecraftian isolation** | **0.000** | 9/9 |
| **body horror and grotesque transformation** | **0.000** | 9/9 |
| **eerie folk horror pagan rituals** | **0.000** | 8/8 |
| **home invasion and paranoia** | **0.000** | 7/8 |

The split is not random. Moods that name **concrete entities or settings** ("haunted
house", "exorcism", "in nature") score well, because those words appear in plot
summaries. Moods describing **tone, register or subgenre** ("campy", "Lovecraftian",
"folk horror", "body horror") score zero, because plot summaries do not contain that
vocabulary at all.

This is the query/document mismatch: users write in *mood* language, the corpus is
embedded from *narrative* summaries, and the two occupy different regions of embedding
space. It is the single clearest lever for improving quality, and it is measurable now.

## Caveat

There is **no before/after comparison**. The previous 21-film corpus was never committed
(`data/` was gitignored) and was overwritten by the rebuild, so its scores cannot be
recovered. Given that corpus consisted mostly of films with "Horror" in the title and
would have contained almost none of the 127 gold titles, its NDCG would necessarily have
been near zero — but that is reasoning, not a measurement, and is not claimed as one.

## The graded benchmark (2026-11-20)

The benchmark above could not decide anything: 15 moods give an interval of
about ±0.06 on NDCG@6, and a film that fits a mood but is missing from its
hand-written list counts as a miss. `scripts/graded_eval.py` is a second
benchmark built to fix both.

- **55 moods** (`eval/moods.json`): the original 15 plus 40 new ones, covering
  subgenres, tones, settings and plain requests ("something scary but not gory
  for a first-timer").
- **Pooled judgments** (`eval/judgments.json`): for each mood, the union of
  what four kinds of ranker return (semantic, the unified blend, keyword only,
  and five noisy production pages) plus the hand-written gold films. 1,271
  mood-film pairs, each graded 2 (squarely fits), 1 (defensible) or 0.
- **The judge was an AI model, not a person.** Four independent passes, one
  per quarter of the moods. The grades are in a readable file, sorted by grade,
  so anyone can correct one. Treat the absolute numbers with that in mind;
  comparisons between rankers judged by the same judge are the sturdier use.

Baseline, `make eval-graded`:

| Arm | NDCG@6 | 95% CI | fits@6 | square@6 |
|-----|-------:|:------:|-------:|---------:|
| semantic, deterministic | 0.6047 | [0.539, 0.671] | 0.691 | 0.430 |
| unified, deterministic | 0.6255 | [0.571, 0.678] | 0.697 | 0.424 |
| unified, with production noise (n=5) | 0.6076 | [0.554, 0.659] | 0.674 | 0.410 |

Paired, over moods:

| Comparison | Mean difference | 95% CI | Verdict |
|------------|----------------:|:------:|---------|
| semantic to unified | +0.0208 | [-0.0135, +0.0544] | indistinguishable |
| unified, deterministic to noisy | -0.0178 | [-0.0333, -0.0012] | noise costs a little |

What this says:

- **The interval on a paired comparison is now about ±0.034**, down from
  ±0.06. A ranking change needs to clear that, not 0.06.
- **Unified is still not shown to beat plain semantic search.** The point
  estimate favours it; the interval includes zero.
- **About 70% of a page fits the mood and about 42% fits it squarely.** That is
  the first absolute statement about quality this project has had.
- **The variety noise costs about 0.018 NDCG**, a real but small price for
  repeat searches differing.
- The weakest moods are tonal or contextual, not topical: "campy fun with lots
  of blood" (0.04), "ghost story that is sad more than scary", "self-aware teen
  horror from the 90s", "extreme, disturbing, hard to watch", "80s practical
  effects gore". This is the missing tone vocabulary of issue #27, seen again.

Limits:

- **Holes.** A film a ranker returns that was never judged scores 0 and is
  counted. A new ranker must be pooled and judged first (`make eval-pool`
  lists what is missing; `pool --weights` adds a variant) or its score is an
  underestimate. The report prints the hole count for every arm.
- **Shallow pools.** The ideal page is built from judged films only. Where
  retrieval is poor the pool is too: "Stephen King adaptation" has two films
  graded 2, though the corpus holds many more that no ranker surfaced.
- **One judge**, with no second opinion to measure agreement against.

### First ablations

`score --compare '<weights>'` re-ranks with different blend weights and pairs
the result against the default, mood by mood. Scoring takes about ten seconds.

| Variant (vs. default unified) | Difference | 95% CI | Holes | Verdict |
|-------------------------------|-----------:|:------:|------:|---------|
| keyword weight 0 | -0.0857 | [-0.1165, -0.0563] | 2 | **worse** |
| popularity weight 0 | -0.0090 | [-0.0267, +0.0088] | 1 | indistinguishable |
| popularity weight 0.4 | -0.0226 | [-0.0435, -0.0011] | 7 | worse, but 7 unjudged films |
| semantic 0.7, popularity 0.1 | -0.0196 | [-0.0443, +0.0056] | 0 | indistinguishable |

- **The keyword (BM25) signal earns its place.** This is the first ranking
  component in this project shown to matter with an interval that excludes
  zero. Without it the unified ranker (0.540) is worse than plain semantic
  search (0.605): popularity, recency and the diversity pass cost relevance,
  and the keyword signal more than buys it back.
- **Popularity is not shown to help or hurt** (the question from issue #34).
  Removing it moves NDCG by less than the interval.
- These are four looks at one benchmark with no holdout. They say which
  components matter; they are not a tuned set of weights. Tuning should split
  the 55 moods into dev and test first.

### Queries with no right answer

`graded_eval.py nulls` compares the best cosine score for the 55 moods against
eight queries a horror corpus cannot serve. They do not separate: "documentary
about the ocean" scores 0.49 and "romantic comedy for date night" 0.43, while
the weakest real mood ("animals attack") scores 0.30 and eight real moods fall
below 0.48. Only nonsense ("asdf qwerty", "best pizza recipe", about 0.2) is
clearly apart. A "we are not sure about this one" notice based on a similarity
threshold would therefore fire on real moods; it was not built.

## Keyword vectors mixed into the stored embeddings (2026-11-20)

The first ranking change the graded benchmark could confirm. Each film's
stored vector is now 70% the vector of its composed document and 30% the
vector of its keywords encoded alone, re-normalised
(`mix_keyword_vectors()` in `app/services/corpus.py`). Issue #28 proposed
scoring fields separately; that was measured first and was far worse.

Retrieval only, against the document-only vectors, before any new judging:

| Variant | NDCG@6 difference | Unjudged films |
|---------|------------------:|---------------:|
| max of title, keywords, overview | -0.149 | 125 |
| mean of the three | -0.116 | 81 |
| keywords only | -0.196 | 151 |
| 90% document + 10% keywords | +0.007 | 0 |
| 80% + 20% | +0.012 | 2 |
| **70% + 30%** | +0.021 | 11 |
| 50% + 50% | -0.018 | 43 |

The 70/30 mix was then implemented, and the 239 films it newly surfaced were
judged blind (the judge did not know which ranker returned them). With no
holes left in either arm:

| Arm | Before | After | Difference | 95% CI | Dev half | Test half |
|-----|-------:|------:|-----------:|:------:|---------:|----------:|
| semantic | 0.5805 | 0.6229 | +0.0424 | [+0.0197, +0.0667] | +0.056 | +0.028 |
| unified | 0.5988 | 0.6390 | +0.0402 | [+0.0148, +0.0667] | +0.028 | +0.053 |
| unified, production noise | 0.5826 | 0.6328 | +0.0502 | [+0.0315, +0.0702] | +0.043 | +0.058 |

- The share of a page that fits the mood rose from about 69% to 72-73%, and
  the share that fits squarely from about 42% to 45-46%.
- 33 moods improved, 17 got worse, 5 did not move. Largest gains: "Stephen
  King adaptation" (+0.35), "cursed videotape or haunted technology" (+0.24),
  "Japanese or Korean ghost horror" (+0.22). Largest losses: "witches and
  covens", "found footage realistic terror" (-0.13 each).
- The "before" numbers are lower than the baseline table above (0.605, 0.626)
  because judging more films found more good ones, which raises the ideal a
  page is measured against. **The tables above are the numbers at the time;
  the current baseline is the "After" column here.**
- On the old 8-mood CI slice the picture is flat: semantic 0.602 to 0.615,
  unified 0.639 to 0.611. Both movements are inside that slice's ±0.06.
- The 30% share was picked on all 55 moods, by looking at four values. The
  dev/test columns show the gain holds on each half, but the share itself was
  not tuned on one half and confirmed on the other.
- Not measured: "more like this" and the taste vector use the same stored
  vectors, so they changed too. Nothing in the benchmark covers them.

## Descriptive tags (2026-11-20)

Issue #27's diagnosis was that TMDB keywords name what is *in* a film
("helicopter", "walrus") and almost never what it is *like*, so tonal requests
have nothing to match. The weakest moods on the graded benchmark were exactly
those. This change supplies the missing words.

- **A fixed vocabulary of 140 tags** in seven facets (subgenre, tone, threat,
  setting, theme, style, audience fit): "slow burn", "folk horror", "campy",
  "very gory", "gateway horror", "single location", "80s horror" and so on.
- **Each of the 500 films carries 1 to 10 of them** (8.4 on average), in
  `data/film_tags.json`, separate from the corpus so a rebuild cannot wipe them.
- **Both were written by an AI model**, from its knowledge of the films. The
  vocabulary was designed by a pass that was not shown the benchmark's moods,
  so the tags were not fitted to the test. The file says who wrote it.

Where the tags go was measured, retrieval only, before any new judging:

| Tags placed in | NDCG@6 difference | Unjudged films |
|----------------|------------------:|---------------:|
| the long document (`embedding_text`) | +0.012 | 32 |
| both the document and the keyword vector | +0.017 | 38 |
| **the keyword vector only** | +0.054 | 14 |
| a third vector of their own (60/20/20) | +0.054 | 26 |
| tags alone | -0.280 | 201 |

So they join the keywords in the second vector and stay out of the document.
Adding them to the text used for keyword matching (BM25) in the unified
strategy was worth a further +0.023 there.

Final comparison, after judging the 146 films the change newly surfaced
(judged from titles and TMDB keywords, without sight of the tags). No holes:

| Arm | Before | After | Difference | 95% CI | Dev half | Test half |
|-----|-------:|------:|-----------:|:------:|---------:|----------:|
| semantic | 0.6163 | 0.6665 | +0.0502 | [+0.0267, +0.0749] | +0.039 | +0.062 |
| unified | 0.6315 | 0.7020 | +0.0705 | [+0.0425, +0.1006] | +0.080 | +0.060 |
| unified, production noise | 0.6252 | 0.6952 | +0.0700 | [+0.0442, +0.0975] | +0.087 | +0.053 |

- With unified, the share of a page that fits rose from 72% to 78%, and the
  share that fits squarely from 45% to 54%.
- 38 moods improved, 10 got worse, 7 did not move. The largest gains are the
  tonal and stylistic requests #27 was about: "80s practical effects gore"
  (0.15 to 0.57), "extreme, disturbing, hard to watch" (0.38 to 0.72),
  "animals attack" (0.36 to 0.69), "self-aware teen horror from the 90s"
  (0.17 to 0.39).
- Largest losses: "Stephen King adaptation" (0.82 to 0.67), "grief and trauma
  as the monster" (0.77 to 0.63), "Japanese or Korean ghost horror" (0.78 to
  0.65). The last is a tagging gap: only two films carry "j-horror".
- **The caveat that matters most:** the tags and the relevance judgments were
  both written by the same kind of model. They share its idea of what, say,
  "campy" means, which favours this change. The vocabulary was blind to the
  moods and the judge was blind to the tags, but a human check of either would
  be worth more than another round of this.
- **A check that does not depend on the AI judge:** the CI slice of the
  original benchmark, 8 moods scored against hand-written gold lists, moved
  from 0.615 to 0.660 (semantic) and from 0.611 to 0.691 (unified). Eight moods
  is a wide interval, but it points the same way.
- Together with the keyword vector, the day's two changes took the unified
  ranker from 0.599 to 0.702 on the judgments as they now stand.

## What visitors were actually served (2026-11-20)

Every number above calls the ranking functions directly. A visitor's request
goes through `recommend_movies_advanced()`, which did one more thing after
ranking: it drew the page at random from the top 18 films, with weights that
fell only to 0.3. Nothing measured that step, because `make eval` and the
graded benchmark both run with `temperature=0`, which skipped it.

Measured through the request path with its randomness on, ten visits per mood,
after judging the 693 films those paths returned (no holes):

| What is served | NDCG@6 | Fits | Fits squarely | Different films over 10 visits |
|----------------|-------:|-----:|--------------:|-------------------------------:|
| semantic, 6 drawn from the top 18 (was live, and the default) | 0.456 | 61% | 30% | 24.4 |
| semantic, 6 drawn from the top 12 | 0.524 | 66% | 37% | 18.5 |
| semantic, 6 drawn from the top 9, steeper weights | 0.594 | 71% | 44% | 13.8 |
| **semantic, top 6, score noise only (now live)** | 0.609 | 72% | 46% | 12.6 |
| semantic, top 6, no randomness at all | 0.632 | 74% | 49% | 6.0 |
| unified, as it was live | 0.556 | 67% | 39% | 23.7 |
| **unified, re-ranking the top 60 (now live, and the default)** | 0.672 | 77% | 52% | 11.3 |

Two defects, one cause:

- **The default page was a third worse than the ranking behind it.** The draw
  bought variety nobody needed: the score noise alone shows about 13 different
  films over ten visits, and a repeat search already demotes what was just
  shown.
- **The unified strategy re-ranked a random 60 of the top 180 films.** It asks
  `recommend_movies_advanced()` for a pool of 60; that function retrieved ten
  times its limit and then ran the same random draw on the pool.

Changes:

- The page is the top of the ranking. Variety comes from the score noise and
  from demoting recently shown films.
- The candidate pool no longer grows with the limit.
- **Unified is the default strategy.** As served it beats semantic by +0.063
  NDCG@6, 95% interval [+0.036, +0.090], the first time the two have been
  told apart. (Deterministically it is +0.035, [+0.003, +0.066].)
- `/api/recommendations?seed=` is now reproducible, as its documentation said.
- `graded_eval.py score` gained "as served" arms that go through the request
  path, so this step cannot go unmeasured again.

The combined effect on the default search page, on today's judgments: NDCG@6
from 0.456 to 0.672, and films that fit the mood from 61% to 77% of the page.

The lesson is the one this document keeps relearning: a number measured with
the randomness switched off describes a system nobody uses.

## Queries that name a film (2026-11-20)

A query such as "like Hereditary but less bleak" used to be embedded as free text. The title's own words dominated: the named film came back first and the rest shared vocabulary with its title. `app/services/title_reference.py` now detects a corpus title in the query, searches with that film's stored vector (blended 70/30 with whatever else the query says), matches keywords against the film's tags and keywords, and leaves the film and the rest of its franchise out of the results. "Franchise" is TMDB's collection plus same-title remakes and entries titled "Title: ..." or "Title 2", because TMDB files prequels and reboots (Alien: Covenant, Halloween 2007) as franchises of their own.

**The cases.** 24 queries in `eval/moods.json` under `title`: bare titles, "like X", "like X but Y", "X meets Y". 721 query-film pairs were pooled from the old ranker, an exclusion-only variant and the new ranker at four blend settings, then graded 0 to 2 by an AI judge blind to the source. The named film itself is graded 0 (the user has seen it) and its sequels and remakes at most 1.

`python scripts/graded_eval.py titles`:

| Arm | NDCG@6 | fits@6 | square@6 |
|---|---|---|---|
| unified as served, title handling off | 0.199 | 34% | 11% |
| unified as served, named film excluded only | 0.273 | 40% | 14% |
| unified as served, film vector (shipped) | 0.637 | 79% | 36% |
| semantic, off | 0.211 | 38% | 9% |
| semantic, film vector | 0.529 | 74% | 29% |

Paired, as served: exclusion alone +0.074 [+0.037, +0.112]; the film vector on top of it +0.364 [+0.277, +0.449]. On unified, every one of the 24 queries scored higher with the film vector than with title handling off.

**What mattered.** The blend share barely does (0.5 to 1.0 all land between 0.647 and 0.651 on unified), so it stays at 0.7. Matching keywords against the named film's own tags and keywords is worth about +0.10 over matching only the rest of the query.

**What this does not show.**

- The rubric grades the named film 0, so part of the gain (the +0.074) is the rubric agreeing with the design. The +0.364 is measured after that is removed.
- I wrote the 24 queries and the detector together, and all 24 are detected. The number says nothing about how often real queries that name a film are recognised. Detection is conservative on purpose: a title counts when it is the whole query, follows a cue such as "like", or opens the query before "but"/"meets"/"vibes". "the thing in the basement" and "halloween" typed alone are treated as moods.
- A wrong detection would replace a mood search with "films like X". None of the 63 mood and null queries in the benchmark, nor the home page's suggestions, triggers it; a test keeps that true.
- "less bleak" and "but funny" are embedded as text. The model does not understand negation or contrast, so the condition nudges the results and no more.
- The judge is an AI model. No person has checked the grades.

## A fitted blend with a tag signal (2026-11-20)

**Where quality was being lost.** For each of the 55 moods, 92% of the films graded 2 are already among the 60 candidates the search retrieves, yet the page scored 0.685. An oracle ordering of the same 60 would score about 0.99. Retrieval was fine; ordering was the problem.

**What was tried**, each fitted on some moods and scored on moods held out of the fit (six repetitions of five folds):

| Ranker | Held-out NDCG@6 | Against the fixed blend |
|---|---|---|
| The fixed blend (semantic .45, keyword .20, popularity .20, recency .05) | 0.679 | |
| Ridge on the same four signals | 0.692 | +0.013 [-0.011, +0.038] |
| Ridge on the four plus a query-to-tag signal | 0.760 | +0.080 [+0.048, +0.114] |
| Gradient boosting on those six | 0.744 | +0.065 [+0.034, +0.096] |
| Gradient boosting on 18 features | 0.729 | +0.050 [+0.006, +0.094] |

Re-weighting what was already there gains nothing, as the earlier weight tuning found. One new signal carries the whole gain, and the simplest model that uses it is the best: more features and trees fit the 55 moods and lose on the held-out ones.

**The signal.** Each of the 140 tags has a one-line meaning in `data/film_tags.json`. Those definitions are embedded once; a film's score is the cosine between the query and the best-matching of its tags' definitions. Taking the best tag beats averaging, and embedding "tag: meaning" beats the tag alone (+0.080 against +0.063).

**What ships.** `DEFAULT_WEIGHTS` is now semantic .30, keyword .25, tags .35, popularity .10, recency 0: a non-negative ridge fit, rounded. For a query that names a film the tag signal is the share of tags the candidate has in common with the named film, which beat both alternatives on the title queries (0.751 as served, against 0.668 using the film's vector and 0.682 with no tag signal).

As served, five draws per query, paired against the previous blend:

| | Before | After | Difference | Page judged a fit |
|---|---|---|---|---|
| 55 moods | 0.677 | 0.738 | +0.061 [+0.026, +0.096] | 77% to 85% |
| 15 hand-written moods | 0.745 | 0.807 | +0.062 [+0.015, +0.116] | 88% to 94% |
| 24 title queries | 0.602 | 0.736 | +0.134 [+0.072, +0.195] | 79% to 89% |

No returned film was left unjudged: 132 new pairs were graded first.

**What this does not show.**

- The shipped weights were fitted on all 55 moods, so 0.738 is in-sample. The honest estimate of the gain is the held-out one, about +0.07, which the as-served difference happens to match.
- An AI model wrote the tags and an AI model graded the films. A signal built from one and scored by the other can agree with itself. The hand-written slice, whose gold films a person chose, moves by the same amount, which is some reassurance and not proof.
- Ten moods got worse by more than 0.05. The largest are literal ones with no tag to match: "cursed videotape or haunted technology" (-0.21), "anthology of short scary stories" (-0.19), "evil clown" (-0.15). The tag signal adds noise when no tag fits, and nothing yet turns its weight down in that case.
- The weakest moods are still weak: "campy fun with lots of blood" 0.10 to 0.19, "ghost story that is sad more than scary" 0.17 to 0.22. Degree and negation need attributes the tags do not have.
- Recency now has no weight and popularity half of what it had. Popular films are shown a little less; that was not measured as a goal in itself.

