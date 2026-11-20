"""The home page's form submits its filter fields even when they are empty."""

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

INDEX = Path(__file__).resolve().parent.parent / "app" / "templates" / "index.html"


def _form_fields() -> list[str]:
    """Names of the inputs inside the home page's search form."""
    html = INDEX.read_text()
    form = html[html.index('id="search-form"') :]
    form = form[: form.index("</form>")]
    return re.findall(r'<(?:input|select|textarea)[^>]*\bname="([^"]+)"', form)


def test_search_as_the_browser_submits_it(client: TestClient) -> None:
    """Every field present, the untouched ones empty: exactly what a browser sends."""
    params = {name: "" for name in _form_fields()}
    params.update(mood="haunted house", strategy="unified", limit="6")
    params.pop("english", None)  # an unticked checkbox is not submitted
    resp = client.get("/recommend", params=params)
    assert resp.status_code == 200, resp.text[:300]
    assert "movie-card" in resp.text or "result" in resp.text


@pytest.mark.parametrize(
    "field", ["min_year", "max_year", "min_rating", "limit", "seed", "english"]
)
def test_each_blank_field_is_ignored(client: TestClient, field: str) -> None:
    resp = client.get("/recommend", params={"mood": "haunted house", field: ""})
    assert resp.status_code == 200, resp.text[:300]


def test_a_filled_filter_still_applies_and_still_validates(client: TestClient) -> None:
    assert (
        client.get("/recommend", params={"mood": "ghosts", "min_year": "1990"}).status_code == 200
    )
    assert client.get("/recommend", params={"mood": "ghosts", "min_year": "abc"}).status_code == 422
    assert (
        client.get("/recommend", params={"mood": "ghosts", "min_year": "1200"}).status_code == 422
    )
    assert client.get("/recommend", params={"mood": "ghosts", "limit": "99"}).status_code == 422
