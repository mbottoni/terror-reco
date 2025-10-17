from __future__ import annotations

from typing import Any

from .strategies.base import RecommenderStrategy
from .strategies.embedding_omdb import EmbeddingOMDbStrategy
from .strategies.keyword_omdb import KeywordOMDbStrategy


def get_strategy(name: str) -> RecommenderStrategy:
	key = (name or "").strip().lower()
	if key in ("embed", "embedding", "tfidf"):
		return EmbeddingOMDbStrategy()
	# default
	return KeywordOMDbStrategy()


async def recommend_movies(mood: str, limit: int = 5, strategy: str = "keyword") -> list[dict[str, Any]]:
	impl = get_strategy(strategy)
	return await impl.recommend(mood=mood, limit=limit)
