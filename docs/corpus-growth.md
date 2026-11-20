# Growing the catalogue

The corpus started as 500 films chosen by vote count. `scripts/grow_corpus.py` adds to it without a rebuild, and `.github/workflows/grow-corpus.yml` runs it every day.

## What a run does

1. **Finds candidates** on TMDB: horror films the corpus does not hold, taken in turn from two lists, the most voted and the best rated. Vote count alone keeps adding recent wide releases (the first hundred it found were all from 2000 or later); the best-rated list is where older and non-English films are.
2. **Fetches each film** from TMDB (plot, cast, keywords, franchise) and its IMDb rating and vote count from OMDb.
3. **Decides.** A film is added when it has an IMDb id, an overview, a poster, an IMDb rating with at least 5,000 votes, a runtime of an hour or more, and is listed as horror and not as family or documentary. Anything else is written to `data/corpus_growth.json` with the reason and not asked about again for 90 days.
4. **Fetches where to watch** for the new films and adds it to `data/watch_providers.json`.
5. **Predicts tags** for them (below).
6. In the workflow: runs the corpus validation and the whole test suite, commits the data files to `main`, and starts the CI pipeline, which builds the image (computing the new embeddings) and deploys it.

```bash
make grow                                              # add up to 40
python scripts/grow_corpus.py --add 40 --dry-run       # look, write nothing
python scripts/grow_corpus.py --add 60 --order rating  # only the best-rated list
```

## The API limits

| Service | Free limit | Cost of a run |
|---|---|---|
| TMDB | No daily quota; about 40 requests a second | One request per list page scanned, two per film |
| OMDb | 1,000 requests a day | One per film looked at, including the ones declined |

A run spends at most `--omdb-budget` OMDb requests (default 200, a fifth of the day's allowance) and stops cleanly if OMDb answers that the quota is gone: the films it could not afford are candidates again the next day. At the default of 40 films a day the catalogue grows by about 280 a week.

## Tags for new films

Tags carry a third of the ranking weight, and the original 500 films were tagged by an AI model that knows them. A new film gets **predicted** tags from `scripts/tag_predictor.py`: the tags its 20 nearest tagged neighbours agree on, blended with how close its own text is to each tag's definition.

Measured by leaving each of the 500 out and predicting it from the rest:

| Tags kept | Share that match the AI pass |
|---|---|
| Those 40% of the neighbour weight agrees on (about 2.5 a film) | 68% |
| Top 3 by blended score | 63% |
| Top 8, as many as the AI pass wrote | 45% |

On the first 95 films it was really used on, tagged properly afterwards, 54% of its tags were confirmed: worse than the estimate, because new films are less like the catalogue than the catalogue is like itself.

So the predictor keeps few: the agreed tags, topped up to three, six at most. Predicted films are listed under `predicted` in `data/film_tags.json`. **They are a stopgap.** Replace them with a proper pass from time to time (the first 150 added films were tagged that way on the day they were added); a test fails when predicted films outnumber the properly tagged ones.

## What growth does to the benchmark

`eval/judgments.json` holds grades for specific films. A new film that reaches a page has no grade and scores zero, so the benchmark's numbers fall as the catalogue grows whether or not the pages got worse. After a batch: `make eval-pool`, grade the listed films, add them, then score. Do not compare a score taken before a batch with one taken after it without doing that.

## Does a bigger catalogue recommend better?

Measured on the first growth, 500 to 650 films (100 from the most-voted list, 55 from the best-rated, 5 removed by the family rule), after grading the 199 films the larger catalogue brought onto pages. As served, five draws per query, the same judgments for both:

| | 500 films | 650 films | Difference |
|---|---|---|---|
| 55 moods, NDCG@6 | 0.720 | 0.729 | +0.009 [-0.015, +0.035] |
| 24 title queries, NDCG@6 | 0.711 | 0.714 | +0.003 [-0.035, +0.045] |
| Moods with fewer than six squarely fitting films | 12 | 9 | |
| Distinct films shown over the 1,650 slots | 310 | 376 | |

No measurable change in page quality, in either direction. That is the result worth having: the earlier attempt at 1,200 films measured worse, on a gold set that could only reward famous films. What growth buys so far is breadth: a fifth of all slots now go to films that were not in the catalogue, in proportion to their share of it, and three thin moods are no longer thin.

## If the schedule does not run

GitHub runs scheduled workflows late and sometimes not at all on a quiet repository (the uptime check's schedule had not fired once on the day this was written). The workflow can be started by hand from the Actions tab or with `gh workflow run grow-corpus.yml`, and `make grow` does the same work locally; a local run needs a commit and push afterwards.
