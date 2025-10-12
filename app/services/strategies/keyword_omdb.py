from __future__ import annotations

from math import log
from typing import Any, Dict, List

from ..omdb_client import get_omdb_client


MOOD_KEYWORDS: Dict[str, List[str]] = {
	"tense": ["suspense", "tension", "edge of your seat"],
	"gory": ["gore", "bloody", "splatter"],
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


def _mood_phrases(mood: str) -> List[str]:
	mood_norm = _normalize(mood)
	phrases: List[str] = []
	for key, kws in MOOD_KEYWORDS.items():
		if key in mood_norm:
			phrases.extend(kws)
	for kws in MOOD_KEYWORDS.values():
		for kw in kws:
			if kw in mood_norm:
				phrases.append(kw)
	if not phrases:
		phrases.append(mood_norm)
	return list(dict.fromkeys(phrases))


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
		phrases = _mood_phrases(mood)
		client = await get_omdb_client()
		ids: List[str] = []
		for phrase in phrases:
			search_q = f"{phrase} horror"
			res = await client.search_titles(search_q, page=1)
			for item in res[:5]:
				imdb_id = item.get("imdbID")
				if isinstance(imdb_id, str):
					ids.append(imdb_id)
		ids = list(dict.fromkeys(ids))[:25]

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

		details_sorted = sorted(details, key=lambda x: x.get("_score", 0.0), reverse=True)
		return [{k: v for k, v in m.items() if k != "_score"} for m in details_sorted[:limit]]
