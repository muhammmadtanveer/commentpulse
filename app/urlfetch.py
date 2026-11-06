"""
CommentPulse — paste-a-URL comment fetching.

What each platform honestly allows through its official API:
  YouTube   — public video comments with a free API key (YOUTUBE_API_KEY). LIVE.
  Reddit    — public post comments; works keyless at low volume, better with
              REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET. LIVE.
  X/Twitter — replies need the paid X API (X_BEARER_TOKEN). Wired, key-ready.
  Facebook / Instagram — Meta does NOT allow fetching by public URL; the
              post must belong to a connected account (Facebook Login flow).
  LinkedIn  — comment access is partner-gated (no public API). Paste/CSV only.
  TikTok    — comment API is research-only/restricted. Paste/CSV only.
"""
import html as html_lib
import os
import re
import time
from datetime import datetime, timezone

import httpx

UA = {"User-Agent": "CommentPulse/1.0 (comment analysis tool)"}
MAX_FETCH = 500


class UrlError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


def detect(url: str):
    """Return (platform, object_id) or raise UrlError with a clear reason."""
    u = (url or "").strip()
    if not u:
        raise UrlError("Paste a post or video URL first.")
    if "youtu.be/" in u:
        vid = re.split(r"[?&#/]", u.split("youtu.be/", 1)[1])[0]
        return "youtube", vid
    m = re.search(r"youtube\.com/(?:watch\?v=|shorts/|live/|embed/)([\w-]{6,})", u)
    if m:
        return "youtube", m.group(1)
    m = re.search(r"reddit\.com/r/[^/]+/comments/([a-z0-9]+)", u)
    if m:
        return "reddit", m.group(1)
    m = re.search(r"(?:twitter\.com|x\.com)/[^/]+/status/(\d+)", u)
    if m:
        return "x", m.group(1)
    if "facebook.com" in u or "fb.watch" in u:
        raise UrlError(
            "Meta does not allow fetching Facebook comments from a public URL — "
            "no tool can legally scrape them. Log in with Facebook above and pick "
            "the post from your Page instead: one click, same result.", 422)
    if "instagram.com" in u:
        raise UrlError(
            "Meta does not allow fetching Instagram comments from a public URL. "
            "Log in with Facebook above (your Instagram must be linked to your "
            "Page) and pick the post there.", 422)
    if "linkedin.com" in u:
        raise UrlError(
            "LinkedIn locks comment access behind a partner programme — there is "
            "no public API for it, for any tool. Use Paste comments or CSV for "
            "LinkedIn threads for now.", 422)
    if "tiktok.com" in u:
        raise UrlError(
            "TikTok's comment API is restricted to approved research accounts. "
            "Use Paste comments or CSV for TikTok threads for now.", 422)
    if "pinterest.com" in u or "pin.it" in u:
        raise UrlError(
            "Pinterest does not expose pin comments to any third-party tool — "
            "there is no public comment API for it. Use Paste comments or CSV "
            "for Pinterest threads for now.", 422)
    raise UrlError("I could not recognise that URL. Supported today: YouTube, "
                   "Reddit, and X links — plus Facebook/Instagram via login.")


def _strip_html(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", text or "")
    text = re.sub(r"<[^>]+>", "", text)
    return html_lib.unescape(text).strip()


# ------------------------------------------------------------------ YouTube
def fetch_youtube(video_id: str) -> list:
    key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    if not key:
        raise UrlError("YouTube fetching needs the free YOUTUBE_API_KEY on the "
                       "server (one-time setup by the site owner). Paste/CSV "
                       "still work.", 501)
    comments, page_token = [], None
    with httpx.Client(timeout=30.0, trust_env=False) as client:
        while len(comments) < MAX_FETCH:
            params = {"part": "snippet", "videoId": video_id,
                      "maxResults": 100, "key": key, "textFormat": "plainText"}
            if page_token:
                params["pageToken"] = page_token
            resp = client.get("https://www.googleapis.com/youtube/v3/commentThreads",
                              params=params)
            if resp.status_code == 403:
                raise UrlError("YouTube refused the request — comments may be "
                               "disabled on that video, or the API quota/key is "
                               "not set up for comments.", 502)
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("items", []):
                top = (item.get("snippet") or {}).get("topLevelComment") or {}
                sn = top.get("snippet") or {}
                text = _strip_html(sn.get("textDisplay", ""))
                if text:
                    comments.append({
                        "text": text, "platform": "youtube",
                        "author": sn.get("authorDisplayName"),
                        "timestamp": sn.get("publishedAt")})
            page_token = data.get("nextPageToken")
            if not page_token:
                break
    return comments


# ------------------------------------------------------------------- Reddit
def _reddit_headers() -> dict:
    cid = os.environ.get("REDDIT_CLIENT_ID", "").strip()
    secret = os.environ.get("REDDIT_CLIENT_SECRET", "").strip()
    if cid and secret:
        try:
            with httpx.Client(timeout=20.0, trust_env=False) as client:
                tok = client.post(
                    "https://www.reddit.com/api/v1/access_token",
                    data={"grant_type": "client_credentials"},
                    auth=(cid, secret), headers=UA).json()
                access = tok.get("access_token")
                if access:
                    return {**UA, "Authorization": f"bearer {access}"}
        except Exception:
            pass
    return UA


def _walk_reddit(children, out):
    for child in children or []:
        if child.get("kind") != "t1":
            continue
        d = child.get("data") or {}
        body = (d.get("body") or "").strip()
        if body and body not in ("[deleted]", "[removed]"):
            ts = d.get("created_utc")
            out.append({
                "text": body, "platform": "reddit",
                "author": d.get("author"),
                "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None})
        replies = d.get("replies")
        if isinstance(replies, dict):
            _walk_reddit(((replies.get("data") or {}).get("children")), out)


def fetch_reddit(post_id: str) -> list:
    headers = _reddit_headers()
    base = "https://oauth.reddit.com" if "Authorization" in headers else "https://www.reddit.com"
    try:
        with httpx.Client(timeout=30.0, trust_env=False, follow_redirects=True) as client:
            resp = client.get(f"{base}/comments/{post_id}.json",
                              params={"raw_json": 1, "limit": 500}, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            raise UrlError(
                "Reddit blocks anonymous requests from servers, so by-link "
                "fetching needs the site owner's free Reddit app credentials "
                "(a one-time setup). Paste/CSV work today.", 503)
        raise UrlError(f"Reddit did not return the comments ({exc.__class__.__name__}). "
                       "Check the link is a public post, or use Paste/CSV.", 502)
    out: list = []
    try:
        children = data[1]["data"]["children"]
    except Exception:
        children = []
    _walk_reddit(children, out)
    return out[:MAX_FETCH]


# ------------------------------------------------------------------------ X
def fetch_x(tweet_id: str) -> list:
    token = os.environ.get("X_BEARER_TOKEN", "").strip()
    if not token:
        raise UrlError("X (Twitter) only sells comment access through its paid "
                       "API — fetching is wired and ready, it switches on when "
                       "the site owner adds the X key. Paste/CSV work today.", 501)
    headers = {"Authorization": f"Bearer {token}"}
    out = []
    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            resp = client.get("https://api.twitter.com/2/tweets/search/recent",
                              params={"query": f"conversation_id:{tweet_id}",
                                      "max_results": 100,
                                      "tweet.fields": "author_id,created_at"},
                              headers=headers)
            resp.raise_for_status()
            for t in resp.json().get("data", []):
                text = (t.get("text") or "").strip()
                if text:
                    out.append({"text": text, "platform": "x",
                                "author": t.get("author_id"),
                                "timestamp": t.get("created_at")})
    except UrlError:
        raise
    except Exception as exc:
        raise UrlError(f"X did not return replies ({exc.__class__.__name__}).", 502)
    return out


FETCHERS = {"youtube": fetch_youtube, "reddit": fetch_reddit, "x": fetch_x}


def fetch_by_url(url: str):
    platform, object_id = detect(url)
    comments = FETCHERS[platform](object_id)
    if not comments:
        raise UrlError(f"{platform.title()} returned 0 comments for that link — "
                       "comments may be disabled or empty.", 404)
    return platform, object_id, comments
