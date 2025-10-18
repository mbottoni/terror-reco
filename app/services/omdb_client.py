from __future__ import annotations

from typing import Any

import httpx

from ..settings import get_settings


class OMDbClient:
	def __init__(self, client: httpx.AsyncClient | None = None) -> None:
		settings = get_settings()
		self._base_url = settings.OMDB_BASE_URL
		self._api_key = settings.OMDB_API_KEY or ""
		self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(12.0, connect=5.0))

	async def _get(self, params: dict[str, Any]) -> dict[str, Any]:
		merged = {"apikey": self._api_key}
		merged.update(params)
		resp = await self._client.get(self._base_url, params=merged)
		resp.raise_for_status()
		data = resp.json()
		# OMDb returns { Response: 'False', Error: '...' }
		if isinstance(data, dict) and data.get("Response") == "False":
			return {}
		return data

	async def search_titles(self, query: str, page: int = 1, *, type_: str = "movie", year: int | None = None) -> list[dict[str, Any]]:
		params: dict[str, Any] = {"s": query, "type": type_, "page": page}
		if year is not None:
			params["y"] = year
		data = await self._get(params)
		results = data.get("Search") if isinstance(data, dict) else None
		return list(results or [])

	async def get_by_id(self, imdb_id: str, *, plot_full: bool = False) -> dict[str, Any]:
		data = await self._get({"i": imdb_id, "plot": ("full" if plot_full else "short")})
		return data or {}

	async def aclose(self) -> None:
		await self._client.aclose()


async def get_omdb_client() -> OMDbClient:
	return OMDbClient()
