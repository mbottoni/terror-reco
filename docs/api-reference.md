# API Reference

All endpoints are served by the FastAPI application at `app/main.py` and its routers.

## Pages (HTML)

### `GET /`

Homepage with the search form, strategy selector, and advanced filters.

**Response:** HTML page.

### `GET /loading`

Loading screen. Displays a spooky animation, then JavaScript redirects to `/recommend` with all query parameters forwarded.

**Query parameters:** Same as `/recommend` (passed through).

### `GET /recommend`

Main recommendation endpoint. Returns an HTML page with the movie results grid.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mood` | string (required) | -- | The user's free-text mood/vibe description |
| `strategy` | string | `"semantic"` | One of: `semantic`, `unified`, `embedding`, `keyword` |
| `min_year` | int | `null` | Minimum release year (1900-2100) |
| `max_year` | int | `null` | Maximum release year (1900-2100) |
| `min_rating` | float | `null` | Minimum IMDb rating (0-10, step 0.1) |
| `limit` | int | `6` | Number of results (1-20) |
| `kind` | string | `"movie"` | `movie`, `series`, or `both` |
| `english` | int | `null` | Set to `1` for English-only results |

**Response:** HTML page with movie cards, detail modal, and feedback buttons.

### `GET /login`

Login form page.

### `POST /login`

Process login.

| Field | Type | Description |
|-------|------|-------------|
| `email` | string | User email |
| `password` | string | User password |
| `csrf_token` | string | CSRF token from the form |

**Response:** Redirect to `/` on success, back to `/login` on failure.

### `GET /register`

Registration form page.

### `POST /register`

Process registration.

| Field | Type | Description |
|-------|------|-------------|
| `email` | string | User email |
| `password` | string | Password (min 8 chars) |
| `csrf_token` | string | CSRF token from the form |

**Response:** Redirect to `/` on success, back to `/register` on failure.

### `GET /logout`

Log out and clear session.

**Response:** Redirect to `/`.

### `GET /history`

User's search history page (requires authentication).

**Response:** HTML page with past searches, or redirect to `/login`.

## API Endpoints (JSON)

### `GET /api/recommend`

JSON API for recommendations (programmatic access).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mood` | string (required) | -- | Free-text mood description |
| `limit` | int | `6` | Number of results (1-20) |
| `strategy` | string | `"keyword"` | `keyword` or `embedding` |

**Response:**

```json
{
  "mood": "haunted house",
  "strategy": "keyword",
  "results": [
    {
      "imdb_id": "tt0167404",
      "title": "The Sixth Sense",
      "overview": "A boy who communicates with spirits...",
      "poster_url": "https://...",
      "year": "1999",
      "vote_average": 8.2,
      "genre": "Drama, Mystery, Thriller",
      "director": "M. Night Shyamalan",
      "actors": "Bruce Willis, Haley Joel Osment, ...",
      "runtime": "107 min",
      "rated": "PG-13",
      "language": "English, Latin, Spanish",
      "country": "United States",
      "awards": "Nominated for 6 Oscars. ..."
    }
  ]
}
```

### `POST /api/feedback`

Submit a like (+1) or dislike (-1) for a movie. Requires authentication.

**Request body (JSON):**

```json
{
  "imdb_id": "tt0167404",
  "title": "The Sixth Sense",
  "rating": 1,
  "mood": "haunted house",
  "strategy": "semantic"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `imdb_id` | string (required) | IMDb ID of the movie |
| `title` | string | Movie title |
| `rating` | int (required) | `1` (like) or `-1` (dislike) |
| `mood` | string | The query that generated this recommendation |
| `strategy` | string | The strategy that was used |

**Behaviour:**
- First submission: creates the feedback entry.
- Same rating again: **toggles off** (removes the feedback).
- Different rating: **switches** (updates from like to dislike or vice versa).

**Response:**

```json
{
  "status": "created",
  "imdb_id": "tt0167404",
  "rating": 1
}
```

Status can be `"created"`, `"updated"`, or `"removed"`.

**Errors:**
- `401` if not authenticated.
- `400` if `imdb_id` is empty or `rating` is not `1` or `-1`.

## Stripe Endpoints

### `GET /stripe/coffee`

"Buy me a coffee" payment page.

### `POST /stripe/create-checkout-session`

Creates a Stripe Checkout session and redirects to Stripe.

### `GET /stripe/success`

Success page after completed payment.

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | Stripe session ID |

### `GET /stripe/cancel`

Cancellation page when user abandons payment.

### `POST /stripe/webhook`

Stripe webhook endpoint. Validates the webhook signature and processes `checkout.session.completed` events.

### `GET /stripe/debug`

Debug endpoint showing Stripe configuration status (only useful in development).
