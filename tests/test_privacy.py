"""The privacy page: public, linked everywhere, and honest about deletion."""

from fastapi.testclient import TestClient


def test_privacy_page_is_public(client: TestClient) -> None:
    resp = client.get("/privacy")
    assert resp.status_code == 200
    assert "<h1" in resp.text and "Privacy" in resp.text


def test_privacy_page_covers_what_google_requires(client: TestClient) -> None:
    text = client.get("/privacy").text
    for needle in ("email address", "Google", "Stripe", "delete the account", "/account/"):
        assert needle in text, needle


def test_footer_links_to_privacy_from_every_page(client: TestClient) -> None:
    for path in ("/", "/auth/login", "/auth/register", "/privacy"):
        assert 'href="/privacy"' in client.get(path).text, path
