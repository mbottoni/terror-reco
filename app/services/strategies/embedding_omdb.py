from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..omdb_client import get_omdb_client


def _normalize(text: str) -> str:
	return (text or "").strip().lower()


def _load_config() -> Dict[str, Any]:
	cfg_path = Path(__file__).resolve().parents[3] / "config_embedding.yaml"
	if not cfg_path.exists():
		return {}
	with cfg_path.open("r", encoding="utf-8") as f:
		return yaml.safe_load(f) or {}


class EmbeddingOMDbStrategy:
	def __init__(self) -> None:
		cfg = _load_config()
		self.min_candidates = int(cfg.get("min_candidates", 30))
		self.max_candidates = int(cfg.get("max_candidates", 120))
		self.search_queries = cfg.get("search_queries", [
			"{mood} horror",
			"horror",
			"scary horror",
			"supernatural horror",
			"slasher horror",
			"zombie horror",
		])
		self.top_k_multiplier = int(cfg.get("top_k_multiplier", 3))
		self.max_features = int(cfg.get("max_features", 5000))
		self.randomize_from_top_k = bool(cfg.get("randomize_from_top_k", True))

	async def _fetch_horror_items(self, mood: str, min_needed: int) -> List[Dict[str, Any]]:
		client = await get_omdb_client()
		ids: List[str] = []
		for q in self.search_queries:
			query = q.format(mood=mood)
			if len(ids) >= self.max_candidates:
				break
			res = await client.search_titles(query, page=1)
			for item in res or []:
				imdb_id = item.get("imdbID")
				if isinstance(imdb_id, str):
					ids.append(imdb_id)
			# try page 2 for generic queries
			if q != "{mood} horror" and len(ids) < self.max_candidates:
				res2 = await client.search_titles(query, page=2)
				for item in res2 or []:
					imdb_id = item.get("imdbID")
					if isinstance(imdb_id, str):
						ids.append(imdb_id)

		ids = list(dict.fromkeys(ids))[: self.max_candidates]

		items: List[Dict[str, Any]] = []
		client = await get_omdb_client()
		for imdb_id in ids:
			d = await client.get_by_id(imdb_id)
			if not d:
				continue
			genre = _normalize(d.get("Genre"))
			if "horror" not in genre:
				continue
			poster = d.get("Poster")
			poster_url = poster if poster and poster != "N/A" else None
			items.append(
				{
					"title": d.get("Title"),
					"overview": d.get("Plot") or "",
					"poster_url": poster_url,
					"release_date": d.get("Released"),
					"vote_average": float(d.get("imdbRating") or 0)
					if (d.get("imdbRating") and d.get("imdbRating") != "N/A")
					else None,
				}
			)
			if len(items) >= max(self.min_candidates, min_needed * 5):
				break
		return items

	async def recommend(self, mood: str, limit: int = 5) -> List[Dict[str, Any]]:
		items = await self._fetch_horror_items(mood=mood, min_needed=limit)
		if not items:
			return []

		plots: List[str] = [_normalize(m.get("overview") or "") for m in items]
		if not any(plots):
			return random.sample(items, k=min(limit, len(items)))

		vectorizer = TfidfVectorizer(stop_words="english", max_features=self.max_features)
		corpus = [_normalize(mood)] + plots
		X = vectorizer.fit_transform(corpus)
		mood_vec = X[0:1]
		plot_vecs = X[1:]
		scores = cosine_similarity(mood_vec, plot_vecs).ravel()

		ranked = sorted(zip(items, scores), key=lambda t: t[1], reverse=True)
		top_k = min(len(ranked), max(10, limit * self.top_k_multiplier))
		pool = [m for (m, s) in ranked[:top_k] if s >= 0]
		if len(pool) < limit:
			pool = [m for (m, _s) in ranked]

		if len(pool) <= limit or not self.randomize_from_top_k:
			return pool[:limit]
		return random.sample(pool, k=limit)
