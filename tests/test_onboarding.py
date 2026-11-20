"""Taste bootstrap: pick films you love right after signing up."""

from __future__ import annotations

import re
from typing import Any

import pytest

import app.onboarding as onboarding
from app.models import MovieFeedback
from app.onboarding import onboarding_films
from app.services.personalization import MIN_LIKES_FOR_TASTE


def _film(n: int, keywords: str, votes: int, collection: int | None = None) -> dict[str, Any]:
    return {
        "imdb_id": f"tt{n:07d}",
        "title": f"Film {n}",
        "year": "1980",
        "keywords": keywords,
        "vote_average": 7.0,
        "imdbVotes": f"{votes:,}",
        "poster_url": f"https://image.tmdb.org/t/p/w500/{n}.jpg",
        "collection_id": collection,
    }


CORPUS = [
    _film(1, "slasher, masked killer", 900_000, collection=10),
    _film(2, "slasher", 800_000, collection=10),  # sequel of film 1
    _film(3, "slasher", 700_000),
    _film(4, "slasher", 600_000),
    _film(5, "slasher", 500_000),
    _film(6, "zombie", 400_000),
    _film(7, "zombie, slasher", 950_000),  # fits two subgenres
    _film(8, "vampire", 300_000),
    _film(9, "romantic comedy", 999_999),  # fits none
    {**_film(10, "zombie", 990_000), "poster_url": None},  # no poster
]


@pytest.fixture(autouse=True)
def _corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(onboarding, "_corpus", lambda: CORPUS)


def _csrf(client: Any, path: str) -> str:
    page = client.get(path).text
    return re.search(r'name="csrf" value="([^"]+)"', page).group(1)  # type: ignore[union-attr]


def _register(client: Any, email: str = "new@example.com") -> Any:
    return client.post(
        "/auth/register",
        data={
            "email": email,
            "password": "password123",
            "confirm_password": "password123",
            "csrf": _csrf(client, "/auth/register"),
        },
        follow_redirects=True,
    )


def _pick(client: Any, ids: list[str]) -> Any:
    return client.post("/onboarding/", data={"imdb_id": ids}, follow_redirects=True)


class TestChoosingTheFilms:
    def test_groups_by_subgenre_most_popular_first(self) -> None:
        groups = {g["label"]: [f["title"] for f in g["films"]] for g in onboarding_films(CORPUS)}
        assert groups["Slasher"] == ["Film 7", "Film 1", "Film 3"]
        assert groups["Vampires"] == ["Film 8"]

    def test_a_film_is_offered_once_even_if_it_fits_two_subgenres(self) -> None:
        titles = [f["title"] for g in onboarding_films(CORPUS) for f in g["films"]]
        assert titles.count("Film 7") == 1
        assert len(titles) == len(set(titles))

    def test_a_franchise_is_offered_once(self) -> None:
        titles = [f["title"] for g in onboarding_films(CORPUS) for f in g["films"]]
        assert "Film 1" in titles and "Film 2" not in titles

    def test_films_without_a_poster_or_a_subgenre_are_left_out(self) -> None:
        titles = [f["title"] for g in onboarding_films(CORPUS) for f in g["films"]]
        assert "Film 9" not in titles and "Film 10" not in titles

    def test_empty_subgenres_are_not_shown(self) -> None:
        labels = [g["label"] for g in onboarding_films(CORPUS)]
        assert labels == ["Slasher", "Zombies", "Vampires"]

    def test_real_corpus_gives_a_usable_grid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from app.services.corpus import load_corpus

        corpus = load_corpus()
        if not corpus:
            pytest.skip("corpus not built in this environment")
        groups = onboarding_films(corpus)
        films = [f for g in groups for f in g["films"]]
        assert len(groups) >= 10 and 25 <= len(films) <= 40
        assert len({f["imdb_id"] for f in films}) == len(films)


class TestFlow:
    def test_registration_lands_on_the_picks(self, client: Any) -> None:
        resp = _register(client)
        assert resp.url.path == "/onboarding/"
        assert "Pick a few films you love" in resp.text
        assert 'name="imdb_id" value="tt0000007"' in resp.text
        assert "Skip for now" in resp.text

    def test_signed_out_is_sent_to_login(self, client: Any) -> None:
        assert client.get("/onboarding/", follow_redirects=False).status_code == 303
        assert client.post("/onboarding/", data={}, follow_redirects=False).status_code == 303

    def test_picks_become_likes_and_switch_personalisation_on(self, client: Any, db: Any) -> None:
        _register(client)
        resp = _pick(client, ["tt0000001", "tt0000006", "tt0000008"])
        assert "tuned to your taste" in resp.text
        rows = db.query(MovieFeedback).all()
        assert sorted(r.imdb_id for r in rows) == ["tt0000001", "tt0000006", "tt0000008"]
        assert all(r.rating == 1 and r.mood is None and r.strategy == "onboarding" for r in rows)
        assert {r.title for r in rows} == {"Film 1", "Film 6", "Film 8"}
        assert len(rows) >= MIN_LIKES_FOR_TASTE

    def test_too_few_picks_says_how_many_more(self, client: Any) -> None:
        _register(client)
        assert "Pick 2 more" in _pick(client, ["tt0000001"]).text

    def test_no_picks_is_fine(self, client: Any, db: Any) -> None:
        _register(client)
        assert _pick(client, []).status_code == 200
        assert db.query(MovieFeedback).count() == 0

    def test_repeats_unknown_ids_and_resubmits_do_not_duplicate(self, client: Any, db: Any) -> None:
        _register(client)
        _pick(client, ["tt0000001", "tt0000001", "tt9999999", "not-an-id"])
        _pick(client, ["tt0000001"])
        assert [r.imdb_id for r in db.query(MovieFeedback).all()] == ["tt0000001"]

    def test_picking_a_previously_disliked_film_turns_it_into_a_like(
        self, client: Any, db: Any
    ) -> None:
        _register(client)
        client.post("/api/feedback", json={"imdb_id": "tt0000003", "title": "Film 3", "rating": -1})
        _pick(client, ["tt0000003"])
        db.expire_all()
        assert [(r.imdb_id, r.rating) for r in db.query(MovieFeedback).all()] == [("tt0000003", 1)]

    def test_already_liked_films_show_as_chosen(self, client: Any) -> None:
        _register(client)
        _pick(client, ["tt0000008"])
        page = client.get("/onboarding/").text
        chosen = re.search(r'value="tt0000008"\s+checked disabled', page)
        assert chosen, "a film already liked should render ticked"

    def test_requires_the_csrf_token(self, client: Any) -> None:
        _register(client)
        resp = client.post(
            "/onboarding/", data={"imdb_id": ["tt0000001"]}, headers={"X-CSRF-Token": "wrong"}
        )
        assert resp.status_code == 403

    def test_account_page_links_back_to_it(self, client: Any) -> None:
        _register(client)
        assert 'href="/onboarding/"' in client.get("/account/").text
