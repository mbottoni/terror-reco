from __future__ import annotations

from typing import Any, Dict, List

from .strategies.base import RecommenderStrategy
from .strategies.keyword_omdb import KeywordOMDbStrategy
from .strategies.embedding_omdb import EmbeddingOMDbStrategy


def get_strategy(name: str) -> RecommenderStrategy:
	key = (name or "").strip().lower()
	if key in ("embed", "embedding", "tfidf"):
		return EmbeddingOMDbStrategy()
	# default
	return KeywordOMDbStrategy()


async def recommend_movies(mood: str, limit: int = 5, strategy: str = "keyword") -> List[Dict[str, Any]]:
	impl = get_strategy(strategy)
	return await impl.recommend(mood=mood, limit=limit)
