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
