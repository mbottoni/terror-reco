import httpx
import pytest
import respx

from app.services.recommender import recommend_movies


def _omdb_search_response(title: str, imdb_id: str, year: str = "2012"):
    """Return a mock OMDb search response containing a single result."""
    return httpx.Response(
        200,
        json={"Search": [{"Title": title, "imdbID": imdb_id, "Type": "movie", "Year": year}]},
    )


def _omdb_detail_response(
    title: str, plot: str, rating: str = "7.2", poster: str = "https://img.example.com/p.jpg"
):
    """Return a mock OMDb detail response."""
    return httpx.Response(
        200,
        json={
            "Title": title,
            "Plot": plot,
            "Poster": poster,
            "Released": "2012-10-31",
            "imdbRating": rating,
            "imdbVotes": "2,000",
            "Genre": "Horror, Thriller",
        },
    )


@pytest.mark.asyncio
@respx.mock
async def test_keyword_strategy(monkeypatch):
    monkeypatch.setenv("OMDB_API_KEY", "dummy")

    # The keyword strategy expands mood into many queries.
    # Use a generic pattern that matches any OMDb search request.
    respx.get("https://www.omdbapi.com/").mock(
        side_effect=lambda req: (
            _omdb_detail_response("Tense Night", "Very tense.")
            if "i" in dict(req.url.params)
            else _omdb_search_response("Tense Night", "tt1111111")
        )
    )

    movies = await recommend_movies("tense", limit=1, strategy="keyword")
    assert len(movies) >= 1
    assert movies[0]["title"] == "Tense Night"


@pytest.mark.asyncio
@respx.mock
async def test_embedding_strategy(monkeypatch):
    monkeypatch.setenv("OMDB_API_KEY", "dummy")

    respx.get("https://www.omdbapi.com/").mock(
        side_effect=lambda req: (
            _omdb_detail_response(
                "Ghost House", "A haunted house with paranormal activities.", poster="N/A"
            )
            if "i" in dict(req.url.params)
            else _omdb_search_response("Ghost House", "tt2222222", year="2014")
        )
    )

    movies = await recommend_movies("paranormal", limit=1, strategy="embedding")
    assert len(movies) >= 1
    assert movies[0]["title"] == "Ghost House"
