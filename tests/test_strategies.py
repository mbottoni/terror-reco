import httpx
import pytest
import respx

from app.services.recommender import recommend_movies


@pytest.mark.asyncio
@respx.mock
async def test_keyword_strategy(monkeypatch):
    monkeypatch.setenv("OMDB_API_KEY", "dummy")
    # search
    respx.get(
        "https://www.omdbapi.com/",
        params={"apikey": "dummy", "s": "tense horror", "type": "movie", "page": 1},
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "Search": [
                    {"Title": "Tense Night", "imdbID": "tt1111111", "Type": "movie", "Year": "2012"}
                ]
            },
        )
    )
    # detail
    respx.get(
        "https://www.omdbapi.com/", params={"apikey": "dummy", "i": "tt1111111", "plot": "short"}
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "Title": "Tense Night",
                "Plot": "Very tense.",
                "Poster": "https://m.media-amazon.com/images/M/xyz.jpg",
                "Released": "2012-10-31",
                "imdbRating": "7.2",
                "imdbVotes": "2,000",
                "Genre": "Horror, Thriller",
            },
        )
    )

    movies = await recommend_movies("tense", limit=1, strategy="keyword")
    assert len(movies) == 1
    assert movies[0]["title"] == "Tense Night"


@pytest.mark.asyncio
@respx.mock
async def test_embedding_strategy(monkeypatch):
    monkeypatch.setenv("OMDB_API_KEY", "dummy")
    # search
    respx.get(
        "https://www.omdbapi.com/",
        params={"apikey": "dummy", "s": "paranormal horror", "type": "movie", "page": 1},
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "Search": [
                    {"Title": "Ghost House", "imdbID": "tt2222222", "Type": "movie", "Year": "2014"}
                ]
            },
        )
    )
    # detail
    respx.get(
        "https://www.omdbapi.com/", params={"apikey": "dummy", "i": "tt2222222", "plot": "short"}
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "Title": "Ghost House",
                "Plot": "A haunted house with paranormal activities.",
                "Poster": "N/A",
                "Released": "2014-10-31",
                "imdbRating": "6.0",
                "imdbVotes": "500",
                "Genre": "Horror",
            },
        )
    )

    movies = await recommend_movies("paranormal", limit=1, strategy="embedding")
    assert len(movies) == 1
    assert movies[0]["title"] == "Ghost House"
