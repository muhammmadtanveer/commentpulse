# CommentPulse — switch on login + URL fetching (owner setup, ~10 min)

The code for Facebook Login, Google Login, Page picking, and URL fetching is
built. It activates when these environment variables exist on the server
(Render → commentpulse service → Environment). **Never put secrets in
GitHub or chat — Render's environment screen only.**

## 1) Render environment variables

| Variable | Where it comes from |
|---|---|
| `META_APP_ID` | Meta app (step 2) |
| `META_APP_SECRET` | Meta app → Settings → Basic → App Secret (Show) |
| `GOOGLE_CLIENT_ID` | Google Cloud OAuth client (step 3) |
| `GOOGLE_CLIENT_SECRET` | same screen |
| `SESSION_SECRET` | any long random string you invent (e.g. 32+ characters) |
| `PUBLIC_BASE_URL` | `https://commentpulse-86rh.onrender.com` |
| `YOUTUBE_API_KEY` | Google Cloud → APIs & Services → Credentials → API key (step 3) |
| `DATABASE_URL` | *Recommended:* free Postgres from neon.tech — without it, users/saved analyses live on Render's temporary disk and reset on redeploys |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | *Optional:* reddit.com/prefs/apps (script app) — Reddit works keyless at low volume |
| `X_BEARER_TOKEN` | *Optional, paid:* developer.x.com — switches on X/Twitter URL fetching |

## 2) Meta app (you already own one: "AI Digital Marketing Assitant")

developers.facebook.com → your app →
- **Add product: Facebook Login** → Settings → *Valid OAuth Redirect URIs*:
  `https://commentpulse-86rh.onrender.com/auth/facebook/callback`
- **Settings → Basic:** App Domains = `commentpulse-86rh.onrender.com`
- Permissions used: `pages_show_list`, `pages_read_engagement`,
  `instagram_basic`, `instagram_manage_comments`, `email`, `public_profile`
- While the app is in **Development mode**, login works for people with a
  role in the app (you + testers you add under App Roles). For **any member
  of the public** to log in, Meta requires **App Review** for those
  permissions + Business Verification — that is the Phase 3 gate.

## 3) Google Cloud (one project does login + YouTube)

console.cloud.google.com →
- Create project → **APIs & Services → Enable:** "YouTube Data API v3"
- **Credentials → Create OAuth client ID (Web application):**
  Authorised redirect URI:
  `https://commentpulse-86rh.onrender.com/auth/google/callback`
  → gives `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET`
- **Credentials → Create API key** → `YOUTUBE_API_KEY`
  (restrict it to YouTube Data API v3)

## 4) After saving env vars

Render redeploys automatically. Then on the live site:
- "Facebook login" / "Google login" buttons (top right) work
- After Facebook login, *My connected accounts* lists your Pages →
  View posts → Analyse (comments fetched with your stored login token;
  the analysis is saved to your account history)
- Paste a YouTube or Reddit URL in the hub → comments fetched + analysed

## Platform reality check (built into the product honestly)

- **Facebook / Instagram:** login-based only — Meta bans URL scraping.
- **YouTube / Reddit:** URL fetch live.
- **X:** URL fetch wired; needs the paid X API key to switch on.
- **LinkedIn / TikTok:** no public comment API exists (partner/research
  gated) — paste/CSV import is the honest route today.
