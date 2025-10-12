from __future__ import annotations

from typing import Any, Dict, List, Protocol


class RecommenderStrategy(Protocol):
	async def recommend(self, mood: str, limit: int = 5) -> List[Dict[str, Any]]:
		...
