"""Publishing a watchlist at an unguessable address, and withdrawing it."""

from __future__ import annotations

import re
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import PublicList, User

EMAIL = "curator@example.com"
PASSWORD = "Str0ngPass!"


def _csrf(client: Any, path: str) -> str:
    page = client.get(path).text
    return re.search(r'name="csrf" value="([^"]+)"', page).group(1)  # type: ignore[union-attr]


def _register(client: Any, email: str = EMAIL) -> None:
    client.post(
        "/auth/register",
        data={
            "email": email,
            "password": PASSWORD,
            "confirm_password": PASSWORD,
            "csrf": _csrf(client, "/auth/register"),
        },
    )


def _save(client: Any, imdb_id: str, title: str) -> None:
    body = {"imdb_id": imdb_id, "title": title, "poster_url": "https://example.com/p.jpg"}
    client.post("/watchlist/toggle", json=body)


def _publish(client: Any, title: str = "Folk horror essentials", description: str = "") -> Any:
    return client.post(
        "/watchlist/publish",
        data={"title": title, "description": description},
        follow_redirects=True,
    )


def _slug(db: Any) -> str:
    db.expire_all()
    return str(db.query(PublicList).one().slug)


@pytest.fixture()
def curator(client: Any) -> Any:
    _register(client)
    _save(client, "tt0000001", "The Wicker Man")
    _save(client, "tt0000002", "Midsommar")
    return client


@pytest.fixture()
def visitor() -> Any:
    """Someone with no account and no cookies."""
    with TestClient(app) as c:
        yield c


class TestPublishing:
    def test_private_until_published(self, curator: Any, db: Any) -> None:
        assert db.query(PublicList).count() == 0
        assert "Share this list" in curator.get("/watchlist/").text

    def test_publishing_creates_an_unguessable_link(self, curator: Any, db: Any) -> None:
        resp = _publish(curator, description="Start here.")
        assert "Your list is public" in resp.text
        slug = _slug(db)
        assert len(slug) >= 12 and not slug.isdigit()
        assert f"/l/{slug}" in resp.text

    def test_slugs_differ_between_users(self, curator: Any, db: Any) -> None:
        _publish(curator)
        with TestClient(app) as raw:
            page = raw.get("/auth/register").text
            csrf = re.search(r'name="csrf" value="([^"]+)"', page).group(1)  # type: ignore[union-attr]
            data = {
                "email": "second@example.com",
                "password": PASSWORD,
                "confirm_password": PASSWORD,
                "csrf": csrf,
            }
            raw.post("/auth/register", data=data)
            token = re.search(r'name="csrf-token" content="([^"]+)"', raw.get("/").text)
            assert token
            headers = {"X-CSRF-Token": token[1]}
            raw.post("/watchlist/toggle", json={"imdb_id": "tt9", "title": "X"}, headers=headers)
            raw.post("/watchlist/publish", data={"title": "Mine"}, headers=headers)
        slugs = [row.slug for row in db.query(PublicList).all()]
        assert len(slugs) == 2 and len(set(slugs)) == 2

    def test_needs_a_title_and_at_least_one_film(self, client: Any, db: Any) -> None:
        _register(client)
        assert "Save at least one film" in _publish(client).text
        _save(client, "tt0000001", "The Wicker Man")
        assert "Give your list a title" in _publish(client, title="   ").text
        assert db.query(PublicList).count() == 0

    def test_publishing_again_edits_and_keeps_the_link(self, curator: Any, db: Any) -> None:
        _publish(curator)
        first = _slug(db)
        assert "was updated" in _publish(curator, title="Renamed", description="New words").text
        assert _slug(db) == first
        row = db.query(PublicList).one()
        assert (row.title, row.description) == ("Renamed", "New words")

    def test_text_is_trimmed_flattened_and_capped(self, curator: Any, db: Any) -> None:
        _publish(curator, title="  A\ttitle\nwith   gaps  " + "x" * 200, description="d" * 500)
        row = db.query(PublicList).one()
        assert row.title.startswith("A title with gaps x") and len(row.title) == 80
        assert len(row.description) == 280

    def test_signed_out_cannot_publish(self, visitor: Any, db: Any) -> None:
        token = re.search(r'name="csrf-token" content="([^"]+)"', visitor.get("/").text)
        assert token
        resp = visitor.post(
            "/watchlist/publish",
            data={"title": "x"},
            headers={"X-CSRF-Token": token[1]},
            follow_redirects=False,
        )
        assert resp.status_code == 303 and db.query(PublicList).count() == 0

    @pytest.mark.parametrize("path", ["/watchlist/publish", "/watchlist/unpublish"])
    def test_requires_the_csrf_token(self, curator: Any, path: str) -> None:
        resp = curator.post(path, data={"title": "x"}, headers={"X-CSRF-Token": "wrong"})
        assert resp.status_code == 403


class TestPublicPage:
    def test_anyone_with_the_link_sees_the_films(self, curator: Any, visitor: Any, db: Any) -> None:
        _publish(curator, description="Start here.")
        resp = visitor.get(f"/l/{_slug(db)}")
        assert resp.status_code == 200
        for text in ("Folk horror essentials", "Start here.", "The Wicker Man", "Midsommar"):
            assert text in resp.text
        assert "A list of 2 films" in resp.text

    def test_it_never_reveals_who_made_it(self, curator: Any, visitor: Any, db: Any) -> None:
        _publish(curator)
        page = visitor.get(f"/l/{_slug(db)}").text
        assert EMAIL not in page and "curator" not in page

    def test_it_is_read_only(self, curator: Any, visitor: Any, db: Any) -> None:
        _publish(curator)
        page = visitor.get(f"/l/{_slug(db)}").text
        assert "removeSaved" not in page and "/watchlist/publish" not in page

    def test_search_engines_are_asked_not_to_index_it(
        self, curator: Any, visitor: Any, db: Any
    ) -> None:
        _publish(curator)
        resp = visitor.get(f"/l/{_slug(db)}")
        assert "noindex" in resp.headers["X-Robots-Tag"]
        assert '<meta name="robots" content="noindex, nofollow">' in resp.text
        # Ordinary pages carry no such tag.
        assert 'name="robots"' not in visitor.get("/").text

    def test_preview_tags_describe_the_list(self, curator: Any, visitor: Any, db: Any) -> None:
        _publish(curator)
        page = visitor.get(f"/l/{_slug(db)}").text
        assert '<meta property="og:title" content="Folk horror essentials">' in page
        assert 'content="Midsommar, The Wicker Man"' in page or (
            'content="The Wicker Man, Midsommar"' in page
        )

    def test_hostile_title_and_description_are_escaped(
        self, curator: Any, visitor: Any, db: Any
    ) -> None:
        _publish(curator, title="<script>alert(1)</script>", description='"><img src=x onerror=1>')
        page = visitor.get(f"/l/{_slug(db)}").text
        assert "<script>alert(1)</script>" not in page
        assert "<img src=x" not in page

    def test_follows_the_watchlist_as_it_changes(self, curator: Any, visitor: Any, db: Any) -> None:
        _publish(curator)
        slug = _slug(db)
        _save(curator, "tt0000003", "The Ritual")
        _save(curator, "tt0000001", "The Wicker Man")  # toggles it off
        page = visitor.get(f"/l/{slug}").text
        assert "The Ritual" in page and "The Wicker Man" not in page

    def test_tmdb_is_credited(self, curator: Any, visitor: Any, db: Any) -> None:
        _publish(curator)
        assert "TMDB" in visitor.get(f"/l/{_slug(db)}").text

    @pytest.mark.parametrize("slug", ["nope", "1", "../../etc/passwd", "a" * 200])
    def test_unknown_addresses_are_404(self, visitor: Any, slug: str) -> None:
        assert visitor.get(f"/l/{slug}").status_code == 404


class TestWithdrawing:
    def test_unpublishing_kills_the_link(self, curator: Any, visitor: Any, db: Any) -> None:
        _publish(curator)
        slug = _slug(db)
        resp = curator.post("/watchlist/unpublish", follow_redirects=True)
        assert "private again" in resp.text
        assert visitor.get(f"/l/{slug}").status_code == 404

    def test_republishing_does_not_revive_the_old_link(
        self, curator: Any, visitor: Any, db: Any
    ) -> None:
        _publish(curator)
        old = _slug(db)
        curator.post("/watchlist/unpublish")
        _publish(curator)
        new = _slug(db)
        assert new != old
        assert visitor.get(f"/l/{old}").status_code == 404
        assert visitor.get(f"/l/{new}").status_code == 200

    def test_deleting_the_account_kills_the_link(self, curator: Any, visitor: Any, db: Any) -> None:
        _publish(curator)
        slug = _slug(db)
        curator.post("/account/delete", data={"current_password": PASSWORD})
        db.expire_all()
        assert db.query(User).count() == 0 and db.query(PublicList).count() == 0
        assert visitor.get(f"/l/{slug}").status_code == 404

    def test_unpublishing_when_nothing_is_published_is_harmless(self, curator: Any) -> None:
        assert curator.post("/watchlist/unpublish", follow_redirects=True).status_code == 200


class TestListLayout:
    def test_list_pages_use_a_grid_class_the_stylesheet_defines(
        self, curator: Any, visitor: Any, db: Any
    ) -> None:
        """Both pages once used `card-grid`, which had no CSS: posters rendered
        at full page width. Guard the class against the stylesheet."""
        _publish(curator)
        css = visitor.get("/static/styles.css").text
        for page in (curator.get("/watchlist/").text, visitor.get(f"/l/{_slug(db)}").text):
            grid = re.search(r'<ul class="([^"]+)">', page)
            assert grid, "list page has no film list"
            assert f".{grid.group(1)}{{" in css, f"no CSS rule for .{grid.group(1)}"

    def test_text_inputs_in_forms_are_styled(self, visitor: Any) -> None:
        css = visitor.get("/static/styles.css").text
        assert '.auth-form input[type="text"]' in css
