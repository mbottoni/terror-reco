"""Security response headers and a small in-process rate limiter.

Both are deliberately dependency-free. The limiter keeps its counters in
memory, which is correct only while the app runs as a single process (it does:
one uvicorn worker behind Caddy). With several workers each would count
separately and the effective limit would multiply; move the counters to a
shared store before scaling out.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# A full Content-Security-Policy is not possible yet: every page carries inline
# <style>, inline <script> and onclick attributes, so script-src/style-src
# would need 'unsafe-inline' and protect nothing. These three directives cost
# nothing and do not touch inline code.
_CSP = "frame-ancestors 'none'; base-uri 'self'; object-src 'none'"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: object, *, hsts: bool) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._hsts = hsts

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        headers = response.headers
        headers.setdefault("X-Content-Type-Options", "nosniff")
        headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        headers.setdefault("X-Frame-Options", "DENY")
        headers.setdefault("Content-Security-Policy", _CSP)
        if self._hsts:
            # Only when served over HTTPS (DEBUG off); on plain-HTTP localhost
            # it would be ignored at best.
            headers.setdefault("Strict-Transport-Security", "max-age=31536000")
        return response


@dataclass(frozen=True)
class Rule:
    """Allow *limit* requests per *window* seconds for one client."""

    name: str
    methods: frozenset[str]
    prefixes: tuple[str, ...]
    limit: int
    window: float = 60.0

    def matches(self, method: str, path: str) -> bool:
        return method in self.methods and path.startswith(self.prefixes)


# Sized for a person, not a script: nobody types 30 searches a minute, and
# each search costs a sentence-transformer forward pass.
DEFAULT_RULES: tuple[Rule, ...] = (
    Rule(
        "auth",
        frozenset({"POST"}),
        # /account/ too: each of its forms is a password check.
        ("/auth/login", "/auth/register", "/account/"),
        limit=10,
    ),
    Rule("checkout", frozenset({"POST"}), ("/stripe/create-checkout-session",), limit=10),
    Rule(
        "search",
        frozenset({"GET"}),
        ("/recommend", "/api/recommendations", "/api/similar/"),
        limit=30,
    ),
    Rule("write", frozenset({"POST"}), ("/api/feedback", "/watchlist/toggle"), limit=60),
)


class RateLimiter:
    """Sliding-window counter per (rule, client)."""

    # Stale keys are swept once the table passes this size, so a flood of
    # distinct addresses cannot grow it without bound.
    _SWEEP_AT = 10_000

    def __init__(
        self,
        rules: tuple[Rule, ...] = DEFAULT_RULES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.rules = rules
        self._clock = clock
        self._hits: dict[tuple[str, str], deque[float]] = {}

    def reset(self) -> None:
        self._hits.clear()

    def check(self, method: str, path: str, client: str) -> float | None:
        """Record a request. Returns seconds to wait if it is over the limit."""
        rule = next((r for r in self.rules if r.matches(method, path)), None)
        if rule is None:
            return None
        now = self._clock()
        hits = self._hits.setdefault((rule.name, client), deque())
        while hits and now - hits[0] >= rule.window:
            hits.popleft()
        if len(hits) >= rule.limit:
            return max(rule.window - (now - hits[0]), 1.0)
        hits.append(now)
        if len(self._hits) > self._SWEEP_AT:
            self._sweep(now)
        return None

    def _sweep(self, now: float) -> None:
        longest = max(r.window for r in self.rules)
        for key in [k for k, v in self._hits.items() if not v or now - v[-1] >= longest]:
            del self._hits[key]


limiter = RateLimiter()

_JSON_PREFIXES = ("/api/", "/watchlist/toggle", "/stripe/")


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Behind Caddy, uvicorn's --proxy-headers makes this the real client.
        client = request.client.host if request.client else "unknown"
        retry_after = limiter.check(request.method, request.url.path, client)
        if retry_after is None:
            return await call_next(request)

        headers = {"Retry-After": str(int(retry_after))}
        message = "Too many requests. Please wait a minute and try again."
        if request.url.path.startswith(_JSON_PREFIXES):
            return JSONResponse({"detail": message}, status_code=429, headers=headers)
        templates = getattr(request.app.state, "templates", None)
        if templates is None:
            return Response(message, status_code=429, headers=headers)
        page: Response = templates.TemplateResponse(
            request,
            "error.html",
            {"icon": "⏳", "heading": "Slow down", "message": message, "user": None},
            status_code=429,
            headers=headers,
        )
        return page
