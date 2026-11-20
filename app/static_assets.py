"""Cache-busting URLs and cache headers for files under ``app/static``.

The stylesheet used to be inlined in every page "for zero-cache reliability",
which also meant it was never cached at all. A content hash in the URL gives
the same guarantee the other way round: when the file changes its URL changes,
so a stale copy cannot be served, and an unchanged file can be cached for a
year.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

STATIC_DIR = Path(__file__).resolve().parent / "static"
_IMMUTABLE = "public, max-age=31536000, immutable"
_HASH_CACHE: dict[str, tuple[float, str]] = {}


def static_url(path: str) -> str:
    """``/static/<path>?v=<content hash>``, recomputed when the file changes."""
    file = STATIC_DIR / path
    try:
        mtime = file.stat().st_mtime
    except OSError:
        return f"/static/{path}"  # missing file: a plain URL that will 404 honestly
    cached = _HASH_CACHE.get(path)
    if cached is None or cached[0] != mtime:
        digest = hashlib.sha256(file.read_bytes()).hexdigest()[:12]
        cached = (mtime, digest)
        _HASH_CACHE[path] = cached
    return f"/static/{path}?v={cached[1]}"


class CachedStaticFiles(StaticFiles):
    """StaticFiles that marks fingerprinted responses as immutable.

    A response is fingerprinted when the URL carries ``?v=`` (see
    :func:`static_url`) or the file sits under ``fonts/``, whose filenames
    include their own version. Everything else keeps Starlette's default
    ETag revalidation.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        query: Any = scope.get("query_string", b"")
        fingerprinted = b"v=" in query or path.startswith("fonts/")
        if fingerprinted and response.status_code in (200, 304):
            response.headers["Cache-Control"] = _IMMUTABLE
        return response
