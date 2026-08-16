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
