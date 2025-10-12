from __future__ import annotations

import random
from math import log
from typing import Any, Dict, List

from ..omdb_client import get_omdb_client


MOOD_KEYWORDS: Dict[str, List[str]] = {
	"gory": ["gore", "bloody", "splatter", "blood"],
	"supernatural": ["supernatural", "ghost", "haunted", "possession", "demonic"],
	"slasher": ["slasher", "serial killer", "stalking"],
	"psychological": ["psychological", "mind-bending", "paranoia"],
	"monster": ["monster", "creature", "alien"],
	"zombie": ["zombie", "undead", "apocalypse"],
	"vampire": ["vampire", "bloodsucker"],
	"witch": ["witch", "witchcraft", "coven"],
	"found footage": ["found footage", "mockumentary"],
	"folk": ["folk horror", "ritual", "pagan"],
	"occult": ["occult", "satanic", "cult"],
	"survival": ["survival", "isolated", "remote"],
	"paranormal": ["paranormal", "haunting"],
	"body": ["body horror", "mutation", "transformation"],
	"lovecraftian": ["lovecraftian", "cosmic horror"],
}


def _normalize(text: str) -> str:
	return text.strip().lower()


def _expand_queries(mood: str) -> List[str]:
	m = _normalize(mood)
	queries: List[str] = [f"{m} horror"]
	# Heuristics to expand search based on common words
	if "blood" in m or "bloody" in m:
		queries += ["gory horror", "gore horror", "bloody horror", "splatter horror"]
	if "fun" in m or "funny" in m or "comedy" in m:
		queries += ["comedy horror", "campy horror"]
	# Generic fallbacks to build a pool
	queries += [
		"horror",
		"scary horror",
		"supernatural horror",
		"slasher horror",
		"zombie horror",
	]
	# De-duplicate while preserving order
	seen = set()
	uniq: List[str] = []
	for q in queries:
		if q not in seen:
			uniq.append(q)
			seen.add(q)
	return uniq


def _score_omdb(detail: Dict[str, Any]) -> float:
	rating_str = (detail.get("imdbRating") or "0").replace("N/A", "0")
	votes_str = (detail.get("imdbVotes") or "0").replace(",", "")
	try:
		rating = float(rating_str)
	except ValueError:
		rating = 0.0
	try:
		votes = int(votes_str)
	except ValueError:
		votes = 0
	return rating * (1 + log(1 + votes))


class KeywordOMDbStrategy:
	async def recommend(self, mood: str, limit: int = 5) -> List[Dict[str, Any]]:
		client = await get_omdb_client()

		ids: List[str] = []
		for q in _expand_queries(mood):
			if len(ids) >= 150:
				break
			res = await client.search_titles(q, page=1)
			for item in res or []:
				imdb_id = item.get("imdbID")
				if isinstance(imdb_id, str):
					ids.append(imdb_id)
			# also try page 2 for generic queries
			if q != f"{_normalize(mood)} horror" and len(ids) < 150:
				res2 = await client.search_titles(q, page=2)
				for item in res2 or []:
					imdb_id = item.get("imdbID")
					if isinstance(imdb_id, str):
						ids.append(imdb_id)

		ids = list(dict.fromkeys(ids))[:150]

		details: List[Dict[str, Any]] = []
		for imdb_id in ids:
			d = await client.get_by_id(imdb_id)
			if not d:
				continue
			genre = (d.get("Genre") or "").lower()
			if "horror" not in genre:
				continue
			poster = d.get("Poster")
			poster_url = poster if poster and poster != "N/A" else None
			details.append(
				{
					"title": d.get("Title"),
					"overview": d.get("Plot"),
					"poster_url": poster_url,
					"release_date": d.get("Released"),
					"vote_average": float(d.get("imdbRating") or 0)
					if (d.get("imdbRating") and d.get("imdbRating") != "N/A")
					else None,
					"_score": _score_omdb(d),
				}
			)
			if len(details) >= max(limit * 6, 30):
				break

		# If still no details, do a final generic horror fetch
		if not details:
			res = await client.search_titles("horror", page=1)
			for item in res or []:
				imdb_id = item.get("imdbID")
				if not isinstance(imdb_id, str):
					continue
				d = await client.get_by_id(imdb_id)
				if not d:
					continue
				genre = (d.get("Genre") or "").lower()
				if "horror" not in genre:
					continue
				poster = d.get("Poster")
				poster_url = poster if poster and poster != "N/A" else None
				details.append(
					{
						"title": d.get("Title"),
						"overview": d.get("Plot"),
						"poster_url": poster_url,
						"release_date": d.get("Released"),
						"vote_average": float(d.get("imdbRating") or 0)
						if (d.get("imdbRating") and d.get("imdbRating") != "N/A")
						else None,
						"_score": _score_omdb(d),
					}
				)
				if len(details) >= limit:
					break

		if not details:
			return []

		details_sorted = sorted(details, key=lambda x: x.get("_score", 0.0), reverse=True)
		# Add a bit of variety: sample from the top pool
		pool = details_sorted[: max(10, limit * 3)]
		if len(pool) <= limit:
			return [{k: v for k, v in m.items() if k != "_score"} for m in pool[:limit]]
		chosen = random.sample(pool, k=limit)
		return [{k: v for k, v in m.items() if k != "_score"} for m in chosen]
