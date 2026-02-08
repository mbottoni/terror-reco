import httpx
import pytest
import respx

from app.services.recommender import recommend_movies
from app.settings import get_settings


@pytest.mark.asyncio
@respx.mock
async def test_recommend_movies_omdb(monkeypatch):
    monkeypatch.setenv("PROVIDER", "omdb")
    monkeypatch.setenv("OMDB_API_KEY", "dummy")

    from app import settings as settings_module

    settings_module.get_settings.cache_clear()
    settings = get_settings()
    base = settings.OMDB_BASE_URL.rstrip("/")

    # The keyword strategy expands the mood into many search queries,
    # so we use a generic route that matches all OMDb requests.
    def _handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        if "i" in params:
            # Detail request
            return httpx.Response(
                200,
                json={
                    "Title": "Gory Night",
                    "Plot": "Very gory.",
                    "Poster": "https://m.media-amazon.com/images/M/abc.jpg",
                    "Released": "2010-10-31",
                    "imdbRating": "7.0",
                    "imdbVotes": "1,000",
                    "Genre": "Horror, Thriller",
                },
            )
        # Search request
        return httpx.Response(
            200,
            json={
                "Search": [
                    {"Title": "Gory Night", "imdbID": "tt1234567", "Type": "movie", "Year": "2010"}
                ]
            },
        )

    respx.get(base + "/").mock(side_effect=_handler)
    respx.get(base).mock(side_effect=_handler)

    movies = await recommend_movies("gory", limit=1)
    assert len(movies) >= 1
    assert movies[0]["title"] == "Gory Night"
    assert movies[0]["poster_url"].startswith("https://m.media-amazon.com/")
