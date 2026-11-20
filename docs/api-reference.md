# API Reference

Every route the application serves. Routes live in `app/main.py` and the
routers it includes.

Three rules apply throughout:

- **CSRF.** Every `POST` except the Stripe webhook needs the session's CSRF
  token: a hidden `csrf` form field, or an `X-CSRF-Token` header for `fetch`.
  Each page carries the token in `<meta name="csrf-token">`. A missing or
  wrong token is `403`.
- **Rate limits** are per client address, per minute: 10 for sign-in,
  registration, password reset and account changes; 10 for checkout; 30 for
  searches; 120 for feedback, watchlist and usage writes; 120 for
  where-to-watch lookups. Over the limit is `429` with `Retry-After`.
- **Validation.** JSON bodies are validated; a malformed body is `422`.

## Pages (HTML)

### `GET /`

Home page: search form, strategy selector, filters, and the account menu.

### `GET /privacy`

What the site stores, who else receives data, and how to export or delete it. Linked from every page's footer. Update `PRIVACY_UPDATED` in `app/main.py` when its text changes.

### `GET /recommend`

The results page.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mood` | string, required | -- | Free-text description of what to watch |
| `strategy` | string | `unified` | `unified` or `semantic`. Anything else falls back to the default |
| `min_year`, `max_year` | int | none | Release year bounds (1900 to 2100) |
| `min_rating` | float | none | Minimum IMDb rating (0 to 10) |
| `limit` | int | `6` | Number of films (1 to 20) |
| `english` | int | none | `1` for English-language films only |
| `seed` | int | none | Makes the result reproducible and turns the randomness off. Used by the eval |
| `ids` | string | none | Comma-separated IMDb ids. Shows exactly these films, with no ranking. This is what a shared link carries |

Filters are applied before ranking. If they leave nothing, the page names the
filter responsible.

### Accounts

| Route | Purpose |
|-------|---------|
| `GET /auth/login`, `POST /auth/login` | Sign in (`email`, `password`). With an email provider configured, a correct password on an unconfirmed account is held and sent a new confirmation link |
| `GET /auth/register`, `POST /auth/register` | Create an account (`email`, `password`, `confirm_password`). With an email provider configured it always lands on `/auth/check-email`, whether or not the address already had an account, and the account must be confirmed before it can sign in. Without one it signs in at once and lands on `/onboarding/` |
| `GET /auth/check-email` | "Check your inbox". The same page for every address |
| `GET /auth/verify/{token}` | Confirm an email address from its emailed link. Links last 24 hours and work once. Does not sign anyone in |
| `POST /auth/verify/resend` | Send the signed-in user a new confirmation link (at most one a minute) |
| `GET /auth/google` | Start "Sign in with Google": redirects to Google with a one-time `state` and a PKCE challenge. `404` unless Google sign-in is configured |
| `GET /auth/google/callback` | Where Google sends the browser back. Checks `state`, exchanges the code server to server, then signs in: the account with that Google id, else the account with that (Google-verified) address, else a new one |
| `POST /auth/logout` | Sign out on this device |
| `POST /auth/logout-all` | Sign out on every device |
| `GET /auth/forgot`, `POST /auth/forgot` | Ask for a password-reset email (`email`). The answer is the same whether or not the address is registered. Unavailable until an email provider is configured |
| `GET /auth/reset/{token}`, `POST /auth/reset/{token}` | Choose a new password (`new_password`, `confirm_password`). Links last 30 minutes and work once |
| `GET /onboarding/`, `POST /onboarding/` | Pick films you love (`imdb_id`, repeated). Each becomes a like |
| `GET /account/` | Account settings |
| `POST /account/password` | Change password (`current_password`, `new_password`, `confirm_password`). Signs out other devices |
| `POST /account/email` | Change email (`new_email`, `current_password`). The new address starts unconfirmed and is sent a link |
| `GET /account/export` | Download your data as JSON |
| `POST /account/delete` | Delete the account and its data (`current_password`) |
| `GET /history/` | Your past searches |
| `GET /watchlist/` | Your saved films |
| `POST /watchlist/publish` | Publish the watchlist, or edit a published one (`title`, `description`) |
| `POST /watchlist/unpublish` | Make it private. The public link stops working for good |
| `GET /l/{slug}` | A published list, read-only, no sign-in. Sent with `noindex` |

Pages that need an account redirect to `/auth/login` when signed out.

## JSON endpoints

### `GET /api/recommendations`

Semantic-strategy results as JSON.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mood` | string, required | -- | Free-text description |
| `limit` | int | `6` | Number of films |
| `seed` | int | none | Reproducible, deterministic result |

```json
{"mood": "folk horror", "count": 6, "results": [{"imdb_id": "tt0070917", "title": "The Wicker Man", "...": "..."}]}
```

`503` if the corpus is missing.

### `GET /api/similar/{imdb_id}`

Films most like the given one, by embedding similarity. `limit` (1 to 20,
default 6). `404` for a film that is not in the corpus.

### `GET /api/providers/{imdb_id}`

Where a film can be streamed, rented or bought, from a stored snapshot. The
region comes from `?region=XX` or, failing that, the `Accept-Language` header.
Stored regions: US, GB, CA, AU, BR, DE, FR, ES.

```json
{"region": "GB", "as_of": "2026-11-20", "stream": ["Amazon Prime Video"],
 "rent": ["Apple TV Store"], "buy": ["Apple TV Store"],
 "link": "https://www.themoviedb.org/movie/530385/watch?locale=GB"}
```

`404` when nothing is known for that film and region. `as_of` is the snapshot
date: availability changes often. Data by JustWatch, via TMDB.

### `POST /api/feedback`

Like or dislike a film. Requires sign-in (`401` otherwise).

```json
{"imdb_id": "tt0070917", "title": "The Wicker Man", "rating": 1, "mood": "folk horror", "strategy": "unified"}
```

`rating` is `1` or `-1`. Sending the same rating again removes it; the other
one switches it. Response: `{"status": "created" | "updated" | "removed", "imdb_id": "...", "rating": 1 | 0 | -1}`.

### `POST /api/set-feedback`

A verdict on a whole result set. Open to signed-out visitors.

```json
{"mood": "folk horror", "strategy": "unified", "imdb_ids": ["tt0070917", "tt8772262"], "verdict": 1}
```

`verdict` is `1` (good picks) or `-1`. Same toggle behaviour as above.
Response: `{"status": "...", "verdict": 1 | 0 | -1}`.

### `POST /api/interaction`

Anonymous usage logging: a visitor opened, clicked through to, or saved a film.

```json
{"result_key": "<64 hex characters from the page>", "imdb_id": "tt0070917", "kind": "open"}
```

`kind` is `open`, `imdb_click`, `save` or `similar_click`. Accepted only for a
result set that this same visitor was shown, and a film in it. Response:
`{"status": "recorded" | "duplicate" | "ignored"}`.

### `POST /watchlist/toggle`

Save a film, or remove it if already saved. Requires sign-in.

```json
{"imdb_id": "tt0070917", "title": "The Wicker Man", "poster_url": "https://image.tmdb.org/t/p/w500/..."}
```

Response: `{"status": "added" | "removed", "imdb_id": "...", "saved": true | false}`.

### `GET /healthz`

Liveness and readiness. `{"status": "ok", "corpus_films": 500, "embeddings_ready": true}`,
or `"degraded"` when the corpus or its embeddings are missing. Always `200`.

## Stripe

| Route | Purpose |
|-------|---------|
| `GET /stripe/coffee` | The donation page. `503` page when Stripe is not configured |
| `POST /stripe/create-checkout-session` | Starts a Checkout session. Returns `{"checkout_url": "..."}` |
| `GET /stripe/success?session_id=...` | Thank-you page, shown only when Stripe confirms the session is paid and belongs to this app |
| `GET /stripe/cancel` | Shown when checkout is abandoned |
| `POST /stripe/webhook` | Stripe events, verified by signature. Records completed payments once, and marks refunds |

## Static files

`/static/...`. The stylesheet is linked with a content hash
(`/static/styles.css?v=...`) and cached for a year; fonts likewise.
