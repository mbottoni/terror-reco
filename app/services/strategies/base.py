from __future__ import annotations

from typing import Any, Protocol


class RecommenderStrategy(Protocol):
	async def recommend(self, mood: str, limit: int = 5) -> list[dict[str, Any]]:
		...
