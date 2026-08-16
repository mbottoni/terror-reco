"""Tests for the TMDB -> corpus mapping and OMDb enrichment overlay.

These cover the field contract the templates and scorers depend on: if the
mapping drifts, the UI silently renders blanks and the popularity signal
degrades, so the assertions here are deliberately field-by-field.
"""

from __future__ import annotations

from typing import Any

from app.services.corpus import (
    apply_omdb_enrichment,
    corpus_fingerprint,
    map_tmdb_to_corpus,
)

IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


def _tmdb_detail(**overrides: Any) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "id": 694,
        "title": "The Shining",
        "original_title": "The Shining",
        "overview": "Jack Torrance accepts a caretaker job at the Overlook Hotel.",
        "poster_path": "/xazWoLealQwEgqZ89MLZklLZD3k.jpg",
        "release_date": "1980-05-23",
        "runtime": 144,
        "vote_average": 8.2,
        "genres": [{"id": 27, "name": "Horror"}, {"id": 53, "name": "Thriller"}],
        "spoken_languages": [{"english_name": "English"}],
        "production_countries": [{"name": "United Kingdom"}, {"name": "United States of America"}],
        "external_ids": {"imdb_id": "tt0081505"},
        "credits": {
            "crew": [
                {"job": "Director", "name": "Stanley Kubrick"},
                {"job": "Screenplay", "name": "Diane Johnson"},
                {"job": "Editor", "name": "Ray Lovejoy"},
            ],
            "cast": [
                {"name": "Jack Nicholson"},
                {"name": "Shelley Duvall"},
                {"name": "Danny Lloyd"},
                {"name": "Scatman Crothers"},
                {"name": "Barry Nelson"},
            ],
        },
        "release_dates": {
            "results": [
                {"iso_3166_1": "GB", "release_dates": [{"certification": "18"}]},
                {
                    "iso_3166_1": "US",
                    "release_dates": [{"certification": ""}, {"certification": "R"}],
                },
            ]
        },
    }
    detail.update(overrides)
    return detail


class TestMapTmdbToCorpus:
    def test_maps_every_field_the_ui_consumes(self) -> None:
        rec = map_tmdb_to_corpus(_tmdb_detail(), image_base=IMAGE_BASE)
        assert rec is not None

        assert rec["imdb_id"] == "tt0081505"
        assert rec["title"] == "The Shining"
        assert rec["year"] == "1980"
        assert rec["release_date"] == "1980-05-23"
        assert rec["poster_url"] == f"{IMAGE_BASE}/xazWoLealQwEgqZ89MLZklLZD3k.jpg"
        assert rec["genre"] == "Horror, Thriller"
        assert rec["director"] == "Stanley Kubrick"
        assert rec["writer"] == "Diane Johnson"
        assert rec["language"] == "English"
        assert rec["country"] == "United Kingdom, United States of America"

    def test_runtime_keeps_omdb_string_format(self) -> None:
        """Templates render runtime verbatim; it must stay "N min", not an int."""
        rec = map_tmdb_to_corpus(_tmdb_detail(), image_base=IMAGE_BASE)
        assert rec is not None
        assert rec["runtime"] == "144 min"

    def test_actors_capped_at_four(self) -> None:
        rec = map_tmdb_to_corpus(_tmdb_detail(), image_base=IMAGE_BASE)
        assert rec is not None
        assert rec["actors"] == "Jack Nicholson, Shelley Duvall, Danny Lloyd, Scatman Crothers"

    def test_us_certification_skips_empty_and_foreign(self) -> None:
        rec = map_tmdb_to_corpus(_tmdb_detail(), image_base=IMAGE_BASE)
        assert rec is not None
        assert rec["rated"] == "R"

    def test_rating_starts_on_tmdb_scale(self) -> None:
        rec = map_tmdb_to_corpus(_tmdb_detail(), image_base=IMAGE_BASE)
        assert rec is not None
        assert rec["vote_average"] == 8.2
        assert rec["rating_source"] == "tmdb"

    def test_drops_film_without_imdb_id(self) -> None:
        """imdb_id keys feedback and dedup -- a record without one is unusable."""
        assert map_tmdb_to_corpus(_tmdb_detail(external_ids={}), image_base=IMAGE_BASE) is None

    def test_drops_film_without_overview(self) -> None:
        """The overview is the embedding input; empty means no semantic signal."""
        assert map_tmdb_to_corpus(_tmdb_detail(overview="   "), image_base=IMAGE_BASE) is None

    def test_missing_optional_fields_become_none_not_crash(self) -> None:
        sparse = {
            "id": 1,
            "title": "Obscure",
            "overview": "Something frightening happens.",
            "external_ids": {"imdb_id": "tt9999999"},
        }
        rec = map_tmdb_to_corpus(sparse, image_base=IMAGE_BASE)
        assert rec is not None
        for field in ("poster_url", "director", "actors", "runtime", "rated", "language"):
            assert rec[field] is None


class TestOmdbEnrichment:
    def test_rating_switches_to_imdb_scale(self) -> None:
        rec = map_tmdb_to_corpus(_tmdb_detail(), image_base=IMAGE_BASE)
        assert rec is not None
        out = apply_omdb_enrichment(
            rec,
            {
                "imdbRating": "8.4",
                "imdbVotes": "1,098,765",
                "Metascore": "66",
                "Awards": "Nominated",
            },
        )
        assert out["vote_average"] == 8.4
        assert out["rating_source"] == "imdb"
        assert out["imdbVotes"] == "1,098,765"
        assert out["Metascore"] == "66"
        assert out["awards"] == "Nominated"

    def test_na_sentinels_do_not_overwrite(self) -> None:
        rec = map_tmdb_to_corpus(_tmdb_detail(), image_base=IMAGE_BASE)
        assert rec is not None
        out = apply_omdb_enrichment(
            rec, {"imdbRating": "N/A", "imdbVotes": "N/A", "Metascore": "N/A"}
        )
        assert out["vote_average"] == 8.2
        assert out["rating_source"] == "tmdb"
        assert out["imdbVotes"] is None

    def test_empty_omdb_response_is_tolerated(self) -> None:
        rec = map_tmdb_to_corpus(_tmdb_detail(), image_base=IMAGE_BASE)
        assert rec is not None
        assert apply_omdb_enrichment(rec, {})["vote_average"] == 8.2

    def test_longer_omdb_plot_replaces_short_overview(self) -> None:
        rec = map_tmdb_to_corpus(_tmdb_detail(overview="Short."), image_base=IMAGE_BASE)
        assert rec is not None
        longer = "A much longer and more detailed plot synopsis with far more signal."
        assert apply_omdb_enrichment(rec, {"Plot": longer})["overview"] == longer

    def test_shorter_omdb_plot_is_ignored(self) -> None:
        rec = map_tmdb_to_corpus(_tmdb_detail(), image_base=IMAGE_BASE)
        assert rec is not None
        original = rec["overview"]
        assert apply_omdb_enrichment(rec, {"Plot": "Tiny."})["overview"] == original


class TestCorpusFingerprint:
    def test_detects_text_change_at_constant_length(self) -> None:
        """The old cache keyed on len(corpus), so edits at equal count went unseen."""
        a = [{"imdb_id": "tt1", "overview": "one"}, {"imdb_id": "tt2", "overview": "two"}]
        b = [{"imdb_id": "tt1", "overview": "one"}, {"imdb_id": "tt2", "overview": "CHANGED"}]
        assert len(a) == len(b)
        assert corpus_fingerprint(a) != corpus_fingerprint(b)

    def test_stable_for_identical_content(self) -> None:
        a = [{"imdb_id": "tt1", "overview": "one"}]
        assert corpus_fingerprint(a) == corpus_fingerprint(list(a))


class TestCorpusIntegrity:
    """Guards against the corpus silently losing data.

    A `--resume` run once overwrote every keyword in the corpus with stale
    checkpoint records. Nothing failed: not a test, not the build validator,
    not CI. These assert on the shipped corpus so that regression is loud.
    """

    def test_shipped_corpus_has_keywords(self) -> None:
        from app.services.corpus import load_corpus

        corpus = load_corpus()
        if not corpus:
            import pytest

            pytest.skip("corpus not built in this environment")
        with_keywords = sum(1 for m in corpus if (m.get("keywords") or "").strip())
        # Keywords carry the tone/subgenre vocabulary worth +0.19 NDCG; losing
        # them is invisible without an explicit check.
        assert with_keywords >= len(corpus) * 0.95, (
            f"only {with_keywords}/{len(corpus)} films have keywords - "
            "the corpus may have been overwritten from a stale build checkpoint"
        )

    def test_embedding_text_is_richer_than_the_plot(self) -> None:
        from app.services.corpus import embedding_text, load_corpus

        corpus = load_corpus()
        if not corpus:
            import pytest

            pytest.skip("corpus not built in this environment")
        sample = corpus[0]
        assert len(embedding_text(sample)) > len(sample.get("overview") or "")
