"""The shared page shell: navigation, account menu, and the home shelf."""

import re

from fastapi.testclient import TestClient


def test_signed_in_pages_show_the_account_menu(client: TestClient) -> None:
    """The nav is shared: every page a signed-in user sees offers the account
    menu, not "Log in". History once rendered without the user and did not."""
    token = re.search(r'name="csrf-token" content="([^"]+)"', client.get("/auth/register").text)
    assert token is not None
    resp = client.post(
        "/auth/register",
        data={
            "email": "nav@example.com",
            "password": "Str0ngPass!",
            "confirm_password": "Str0ngPass!",
            "csrf": token.group(1),
        },
    )
    assert resp.status_code == 200
    for path in (
        "/",
        "/history/",
        "/watchlist/",
        "/account/",
        "/privacy",
        "/recommend?mood=ghosts",
    ):
        page = client.get(path).text
        assert 'class="avatar-btn"' in page, path
        assert 'href="/auth/login"' not in page, path


def test_home_page_offers_a_shelf_of_films_to_start_from(client: TestClient) -> None:
    page = client.get("/").text
    assert page.count('class="shelf-item"') == 8
    assert "/recommend?mood=like%20" in page
