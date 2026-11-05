# CommentPulse

Comment intelligence for Facebook & Instagram: paste in comments and
CommentPulse tells you **what people are talking about**, **what keeps
repeating** (the overlap), the **lub-e-lubab (gist)** of the whole
conversation, and **what is likely to come next**.

Built by UpSkillz Solutions. Spec: `~/workspace/comment-intel-product/SPEC.md`.

**Live demo:** https://commentpulse-86rh.onrender.com (Render free tier — sleeps when idle, first load can take ~50s to wake)

## Run it (Demo Mode)

```bash
cd ~/workspace/commentpulse
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --port 8100
```

Open http://localhost:8100 — the dashboard loads in Demo Mode and analyses
the bundled sample dataset (`data/sample_comments.csv`, ~148 comments for a
**fictional** pet grooming business, "Paws & Shine Grooming"). No internet
connection or account is needed; the dashboard uses no external assets.

## What it computes (all numbers are computed — nothing is invented)

- **Topics** — TF-IDF + KMeans clustering; the cluster count adapts to the
  data (~1 per 15 comments, min 3, max 10). Each cluster gets a human label
  (price, booking, complaints, praise, hours & location, services, spam…)
  matched from its content, and the raw top TF-IDF terms are always shown
  next to the label.
- **Overlap** — near-duplicate detection via TF-IDF cosine similarity:
  groups of comments asking effectively the same question, with counts and
  example comments.
- **Sentiment** — VADER per comment; overall and per-topic
  positive/neutral/negative splits.
- **Gist (lub-e-lubab)** — a short templated summary built only from the
  computed clusters, sentiment and overlap, with real example comments
  cited. No LLM in Phase 1, so nothing can be hallucinated.
- **Predictions** — week-over-week topic share changes ("rising topics"),
  sentiment direction, the repeated questions most likely to flood the next
  post, and content ideas derived from the top overlap group. Every
  prediction is labelled **ESTIMATE**.

## Endpoints

| Route | What it does |
|---|---|
| `GET /` | Dashboard (demo analysis on load) |
| `GET /api/demo` | Analysis JSON for the sample dataset |
| `POST /api/import/text` | Analyse JSON: `{"comments": [{"text": "...", "timestamp": "...", "platform": "facebook"}]}` |
| `POST /api/import/csv` | Upload a CSV with columns `text,timestamp,platform,author` |
| `GET /report/demo` | Printable HTML report of the demo analysis |
| `GET /api/demo/whatsapp` | Short plain-text summary (gist + top topics + top repeated question) |
| `POST /api/meta/fetch` | Fetch live comments via Meta Graph API — guarded, see below |
| `GET /api/meta/status` | Whether a Meta token is configured |

## Adding live Meta data later

Set a token in the environment before starting the server:

```bash
export META_ACCESS_TOKEN="<token with pages_read_engagement / instagram permissions>"
.venv/bin/uvicorn app.main:app --port 8100
```

Then `POST /api/meta/fetch` with
`{"platform": "facebook", "object_id": "<post_id>"}` or
`{"platform": "instagram", "object_id": "<media_id>"}` returns the comments
plus their analysis. Without the token the endpoint returns **501** with a
clear "not configured" message — CommentPulse never fakes data.
Note: reading other people's Pages/accounts requires Meta app review;
own and client-granted access is the realistic route (see spec).

## Phase 2 notes (from the spec)

- Predictions module expansion + YouTube (Data API) and TikTok imports.
- Auto-reply suggestions for the top repeated questions.
- Trend comparison across posts/weeks; PDF export polish.
- Longer term (Phase 3): multi-client SaaS with white-label UpSkillz reports.

## Honest limits

- Comment volume matters: under 20 comments the clusters are thin and the
  report says so.
- Predictions are trend-based estimates, always labelled ESTIMATE, never
  presented as facts.
