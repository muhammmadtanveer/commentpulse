"""
CommentPulse — Meta Graph API fetcher (Phase 1 skeleton).

Reads the access token from the META_ACCESS_TOKEN environment variable.
If it is not set, every fetch raises NotConfigured — the app never fakes
data and never attempts a live call without a token.
"""
import os

import httpx

GRAPH_VERSION = "v21.0"
GRAPH_BASE = f"https://graph.facebook.com/{GRAPH_VERSION}"


class NotConfigured(Exception):
    """Raised when META_ACCESS_TOKEN is not set in the environment."""


def _token(explicit: str | None = None) -> str:
    token = (explicit or "").strip() or os.environ.get("META_ACCESS_TOKEN", "").strip()
    if not token:
        raise NotConfigured(
            "Meta API not configured: set the META_ACCESS_TOKEN environment "
            "variable with a valid Facebook/Instagram access token to fetch "
            "live comments."
        )
    return token


def _get(path: str, params: dict, token: str | None = None) -> dict:
    params = {**params, "access_token": _token(token)}
    with httpx.Client(timeout=30.0, trust_env=False) as client:
        resp = client.get(f"{GRAPH_BASE}/{path}", params=params)
        resp.raise_for_status()
        return resp.json()


def _normalise(item: dict, platform: str) -> dict:
    author = item.get("username")
    if not author and isinstance(item.get("from"), dict):
        author = item.get("from", {}).get("name")
    return {
        "text": item.get("text") or item.get("message") or "",
        "timestamp": item.get("created_time") or item.get("timestamp"),
        "platform": platform,
        "author": author,
    }


def fetch_facebook_post_comments(post_id: str, limit: int = 100, token: str | None = None) -> list:
    """Fetch comments on a Facebook Page post via the Graph API."""
    data = _get(
        f"{post_id}/comments",
        {"fields": "message,created_time,from", "limit": limit},
        token=token,
    )
    return [_normalise(c, "facebook") for c in data.get("data", []) if (c.get("message") or "").strip()]


def fetch_instagram_media_comments(media_id: str, limit: int = 100, token: str | None = None) -> list:
    """Fetch comments on an Instagram media item via the Graph API."""
    data = _get(
        f"{media_id}/comments",
        {"fields": "text,timestamp,username", "limit": limit},
        token=token,
    )
    return [_normalise(c, "instagram") for c in data.get("data", []) if (c.get("text") or "").strip()]


def is_configured() -> bool:
    return bool(os.environ.get("META_ACCESS_TOKEN", "").strip())


# ---- helpers for logged-in (connected account) flows -----------------------

def graph_get(path: str, token: str, params: dict | None = None) -> dict:
    """GET against the Graph API with an explicit token (user or page)."""
    return _get(path, params or {}, token=token)


def get_user_pages(user_token: str) -> list:
    """Pages the user manages, with page tokens and linked IG accounts.

    Page access tokens stay server-side; only safe display fields are
    returned to callers that pass data to the browser.
    """
    data = _get(
        "me/accounts",
        {"fields": "id,name,category,access_token,instagram_business_account{id,username,name}",
         "limit": 100},
        token=user_token,
    )
    return data.get("data", [])


def get_page_posts(page_id: str, page_token: str, limit: int = 25) -> list:
    data = _get(
        f"{page_id}/posts",
        {"fields": "id,message,created_time,comments.limit(0).summary(true)",
         "limit": limit},
        token=page_token,
    )
    return data.get("data", [])


def get_ig_media(ig_user_id: str, token: str, limit: int = 25) -> list:
    data = _get(
        f"{ig_user_id}/media",
        {"fields": "id,caption,media_type,timestamp,comments_count,permalink",
         "limit": limit},
        token=token,
    )
    return data.get("data", [])
