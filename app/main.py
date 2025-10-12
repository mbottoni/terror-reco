from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .settings import get_settings
from .services.recommender import recommend_movies


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title=get_settings().APP_NAME, debug=get_settings().DEBUG)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
	return templates.TemplateResponse("index.html", {"request": request})


@app.get("/recommend", response_class=HTMLResponse)
async def ui_recommendations(request: Request, mood: str = Query(..., min_length=1)) -> HTMLResponse:
	movies = await recommend_movies(mood=mood, limit=5)
	return templates.TemplateResponse(
		"results.html", {"request": request, "mood": mood, "movies": movies}
	)


@app.get("/api/recommendations")
async def api_recommendations(mood: str = Query(..., min_length=1), limit: int = 5) -> Dict[str, Any]:
	try:
		movies = await recommend_movies(mood=mood, limit=limit)
	except HTTPException:
		raise
	except Exception as exc:  # pragma: no cover
		raise HTTPException(status_code=500, detail=str(exc))

	return {"mood": mood, "count": len(movies), "results": movies}
