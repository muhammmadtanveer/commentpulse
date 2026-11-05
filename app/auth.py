"""
CommentPulse — accounts: Facebook Login, Google Login, connected pages.

Login activates only when the matching env vars are set on the server:
  META_APP_ID / META_APP_SECRET           (his Meta app — e.g. his FYP app)
  GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET (Google Cloud OAuth client)
  SESSION_SECRET                          (any long random string)
  PUBLIC_BASE_URL                         (e.g. https://commentpulse-86rh.onrender.com)

Flow the user asked for: log in with Facebook -> the app lists the Pages
they manage -> pick a Page -> pick a post -> comments are fetched with the
stored token and analysed. No token is ever shown in the browser or chat.
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import urllib.parse

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from app import db
from app.analyze import analyze
from app.meta_api import (
    fetch_facebook_post_comments,
    fetch_instagram_media_comments,
    get_ig_media,
    get_page_posts,
    get_user_pages,
    graph_get,
)

FB_SCOPES = ("public_profile,email,pages_show_list,pages_read_engagement,"
             "instagram_basic,instagram_manage_comments")
GOOGLE_SCOPES = ("openid email profile "
                 "https://www.googleapis.com/auth/youtube.readonly")
GRAPH = "https://graph.facebook.com/v21.0"

_page_token_cache: dict = {}


# ---------------------------------------------------------------- sessions
def _secret() -> bytes:
    return os.environ.get("SESSION_SECRET", "dev-secret-change-me").encode()


def _sign(payload: dict) -> str:
    raw = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    sig = hmac.new(_secret(), raw.encode(), hashlib.sha256).hexdigest()
    return f"{raw}.{sig}"


def _unsign(value: str):
    try:
        raw, sig = value.rsplit(".", 1)
        expected = hmac.new(_secret(), raw.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(base64.urlsafe_b64decode(raw.encode()))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


def current_user(request: Request):
    payload = _unsign(request.cookies.get("cp_session", ""))
    if not payload:
        return None
    return db.get_user(payload.get("uid", ""))


def _session_cookie(resp, user_id: str):
    resp.set_cookie("cp_session", _sign({"uid": user_id, "exp": time.time() + 30 * 86400}),
                    max_age=30 * 86400, httponly=True, samesite="lax")
    return resp


def _base_url(request: Request) -> str:
    env = os.environ.get("PUBLIC_BASE_URL", "").strip().rstrip("/")
    return env or str(request.base_url).rstrip("/")


def _not_configured(what: str):
    return JSONResponse(status_code=501, content={
        "error": f"{what} login is not switched on yet — the server is missing its app credentials. "
                 "The site owner sets them once (see SETUP_AUTH.md); paste / CSV / URL analysis still work."})


def _meta_error(exc: Exception):
    if isinstance(exc, httpx.HTTPStatusError):
        detail = ""
        try:
            detail = exc.response.json().get("error", {}).get("message", "")
        except Exception:
            pass
        return JSONResponse(status_code=502, content={
            "error": "Meta rejected the request" + (f": {detail}" if detail else
                     f" (HTTP {exc.response.status_code}). Your login may have expired — log in with Facebook again.")})
    return JSONResponse(status_code=502, content={
        "error": f"Could not reach Meta ({exc.__class__.__name__}). Try again in a moment."})


# ------------------------------------------------------------------- models
class FBAnalyze(BaseModel):
    page_id: str
    post_id: str


class IGAnalyze(BaseModel):
    ig_id: str
    media_id: str


class SaveAnalysis(BaseModel):
    label: str
    analysis: dict


# ------------------------------------------------------------------- routes
def register(app):

    # ---------------- Facebook Login ----------------
    @app.get("/auth/facebook/login")
    def fb_login(request: Request):
        app_id = os.environ.get("META_APP_ID", "").strip()
        if not app_id:
            return _not_configured("Facebook")
        state = secrets.token_urlsafe(24)
        redirect_uri = f"{_base_url(request)}/auth/facebook/callback"
        url = ("https://www.facebook.com/v21.0/dialog/oauth?" + urllib.parse.urlencode({
            "client_id": app_id, "redirect_uri": redirect_uri,
            "scope": FB_SCOPES, "state": state, "response_type": "code"}))
        resp = RedirectResponse(url)
        resp.set_cookie("cp_state", state, max_age=600, httponly=True, samesite="lax")
        return resp

    @app.get("/auth/facebook/callback")
    def fb_callback(request: Request, code: str = "", state: str = ""):
        app_id = os.environ.get("META_APP_ID", "").strip()
        app_secret = os.environ.get("META_APP_SECRET", "").strip()
        if not app_id or not app_secret:
            return _not_configured("Facebook")
        if not code or state != request.cookies.get("cp_state", ""):
            return JSONResponse(status_code=400, content={"error": "Facebook login was cancelled or the state check failed. Try again."})
        redirect_uri = f"{_base_url(request)}/auth/facebook/callback"
        try:
            with httpx.Client(timeout=30.0, trust_env=False) as client:
                short = client.get(f"{GRAPH}/oauth/access_token", params={
                    "client_id": app_id, "client_secret": app_secret,
                    "redirect_uri": redirect_uri, "code": code}).json()
                short_token = short.get("access_token")
                if not short_token:
                    return JSONResponse(status_code=502, content={"error": f"Facebook did not return a token: {short.get('error', {}).get('message', 'unknown error')}"})
                long_resp = client.get(f"{GRAPH}/oauth/access_token", params={
                    "grant_type": "fb_exchange_token", "client_id": app_id,
                    "client_secret": app_secret, "fb_exchange_token": short_token}).json()
                token = long_resp.get("access_token", short_token)
                expires_in = long_resp.get("expires_in")
                me = client.get(f"{GRAPH}/me", params={
                    "fields": "id,name,email,picture", "access_token": token}).json()
        except Exception as exc:
            return _meta_error(exc)
        avatar = ((me.get("picture") or {}).get("data") or {}).get("url")
        uid = db.upsert_user("facebook", str(me.get("id", "")), me.get("name"),
                             me.get("email"), avatar)
        db.save_token(uid, "facebook", token,
                      time.time() + expires_in if expires_in else None)
        resp = RedirectResponse("/")
        _session_cookie(resp, uid)
        resp.delete_cookie("cp_state")
        return resp

    # ---------------- Google Login ----------------
    @app.get("/auth/google/login")
    def google_login(request: Request):
        client_id = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
        if not client_id:
            return _not_configured("Google")
        state = secrets.token_urlsafe(24)
        redirect_uri = f"{_base_url(request)}/auth/google/callback"
        url = ("https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode({
            "client_id": client_id, "redirect_uri": redirect_uri,
            "response_type": "code", "scope": GOOGLE_SCOPES,
            "state": state, "access_type": "offline", "prompt": "consent"}))
        resp = RedirectResponse(url)
        resp.set_cookie("cp_state", state, max_age=600, httponly=True, samesite="lax")
        return resp

    @app.get("/auth/google/callback")
    def google_callback(request: Request, code: str = "", state: str = ""):
        client_id = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
        client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
        if not client_id or not client_secret:
            return _not_configured("Google")
        if not code or state != request.cookies.get("cp_state", ""):
            return JSONResponse(status_code=400, content={"error": "Google login was cancelled or the state check failed. Try again."})
        redirect_uri = f"{_base_url(request)}/auth/google/callback"
        try:
            with httpx.Client(timeout=30.0, trust_env=False) as client:
                tok = client.post("https://oauth2.googleapis.com/token", data={
                    "code": code, "client_id": client_id, "client_secret": client_secret,
                    "redirect_uri": redirect_uri, "grant_type": "authorization_code"}).json()
                access = tok.get("access_token")
                if not access:
                    return JSONResponse(status_code=502, content={"error": f"Google did not return a token: {tok.get('error_description', 'unknown error')}"})
                me = client.get("https://openidconnect.googleapis.com/v1/userinfo",
                                headers={"Authorization": f"Bearer {access}"}).json()
        except Exception as exc:
            return JSONResponse(status_code=502, content={"error": f"Could not complete Google login ({exc.__class__.__name__}). Try again."})
        uid = db.upsert_user("google", str(me.get("sub", "")), me.get("name"),
                             me.get("email"), me.get("picture"))
        exp = time.time() + tok.get("expires_in", 3600)
        db.save_token(uid, "google", access, exp)
        if tok.get("refresh_token"):
            db.save_token(uid, "google_refresh", tok["refresh_token"], None)
        resp = RedirectResponse("/")
        _session_cookie(resp, uid)
        resp.delete_cookie("cp_state")
        return resp

    @app.post("/auth/logout")
    def logout():
        resp = JSONResponse({"ok": True})
        resp.delete_cookie("cp_session")
        return resp

    # ---------------- account + connected pages ----------------
    @app.get("/api/me")
    def me(request: Request):
        user = current_user(request)
        if not user:
            return {"user": None, "facebook_connected": False}
        return {
            "user": {"name": user.get("name"), "email": user.get("email"),
                     "avatar": user.get("avatar"), "provider": user.get("provider")},
            "facebook_connected": bool(db.get_token(user["id"], "facebook")),
        }

    def _require_fb(request: Request):
        user = current_user(request)
        if not user:
            return None, None, JSONResponse(status_code=401, content={"error": "Log in first."})
        token = db.get_token(user["id"], "facebook")
        if not token:
            return user, None, JSONResponse(status_code=401, content={
                "error": "Facebook is not connected for this account. Log in with Facebook to see your Pages."})
        return user, token, None

    def _page_token(user_id: str, user_token: str, page_id: str):
        cached = _page_token_cache.get(user_id, {}).get(page_id)
        if cached:
            return cached
        pages = get_user_pages(user_token)
        _page_token_cache[user_id] = {p["id"]: p.get("access_token") for p in pages}
        return _page_token_cache[user_id].get(page_id)

    @app.get("/api/facebook/pages")
    def fb_pages(request: Request):
        user, token, err = _require_fb(request)
        if err:
            return err
        try:
            pages = get_user_pages(token)
        except Exception as exc:
            return _meta_error(exc)
        _page_token_cache[user["id"]] = {p["id"]: p.get("access_token") for p in pages}
        out = []
        for p in pages:
            ig = p.get("instagram_business_account") or {}
            out.append({
                "id": p["id"], "name": p.get("name"), "category": p.get("category"),
                "instagram": {"id": ig.get("id"), "username": ig.get("username")} if ig.get("id") else None,
            })
        return {"pages": out}

    @app.get("/api/facebook/pages/{page_id}/posts")
    def fb_posts(request: Request, page_id: str):
        user, token, err = _require_fb(request)
        if err:
            return err
        try:
            page_token = _page_token(user["id"], token, page_id)
            if not page_token:
                return JSONResponse(status_code=404, content={"error": "That Page was not found in your account."})
            posts = get_page_posts(page_id, page_token)
        except Exception as exc:
            return _meta_error(exc)
        return {"posts": [{
            "id": p["id"],
            "message": (p.get("message") or "")[:140],
            "created_time": p.get("created_time"),
            "comments_count": ((p.get("comments") or {}).get("summary") or {}).get("total_count", 0),
        } for p in posts]}

    @app.post("/api/facebook/analyze")
    def fb_analyze(request: Request, req: FBAnalyze):
        user, token, err = _require_fb(request)
        if err:
            return err
        try:
            page_token = _page_token(user["id"], token, req.page_id) or token
            comments = fetch_facebook_post_comments(req.post_id, 500, token=page_token)
        except Exception as exc:
            return _meta_error(exc)
        if not comments:
            return JSONResponse(status_code=404, content={
                "error": "That post has 0 comments Meta will return. Pick a post with comments."})
        result = analyze(comments)
        label = f"Facebook post {req.post_id} ({len(comments)} comments)"
        db.save_analysis(user["id"], label, result)
        return {"fetched": len(comments), "label": label, "analysis": result}

    @app.get("/api/instagram/{ig_id}/media")
    def ig_media(request: Request, ig_id: str):
        user, token, err = _require_fb(request)
        if err:
            return err
        try:
            media = get_ig_media(ig_id, token)
        except Exception as exc:
            return _meta_error(exc)
        return {"media": [{
            "id": m["id"], "caption": (m.get("caption") or "")[:140],
            "timestamp": m.get("timestamp"), "comments_count": m.get("comments_count", 0),
            "permalink": m.get("permalink"),
        } for m in media]}

    @app.post("/api/instagram/analyze")
    def ig_analyze(request: Request, req: IGAnalyze):
        user, token, err = _require_fb(request)
        if err:
            return err
        try:
            comments = fetch_instagram_media_comments(req.media_id, 500, token=token)
        except Exception as exc:
            return _meta_error(exc)
        if not comments:
            return JSONResponse(status_code=404, content={
                "error": "That media has 0 comments Meta will return. Pick one with comments."})
        result = analyze(comments)
        label = f"Instagram media {req.media_id} ({len(comments)} comments)"
        db.save_analysis(user["id"], label, result)
        return {"fetched": len(comments), "label": label, "analysis": result}

    # ---------------- saved analyses ----------------
    @app.post("/api/analyses")
    def save_analysis(request: Request, req: SaveAnalysis):
        user = current_user(request)
        if not user:
            return JSONResponse(status_code=401, content={"error": "Log in to save analyses."})
        aid = db.save_analysis(user["id"], req.label, req.analysis)
        return {"id": aid}

    @app.get("/api/analyses")
    def list_analyses_route(request: Request):
        user = current_user(request)
        if not user:
            return JSONResponse(status_code=401, content={"error": "Log in first."})
        return {"analyses": db.list_analyses(user["id"])}

    @app.get("/api/analyses/{analysis_id}")
    def get_analysis_route(request: Request, analysis_id: str):
        user = current_user(request)
        if not user:
            return JSONResponse(status_code=401, content={"error": "Log in first."})
        row = db.get_analysis(user["id"], analysis_id)
        if not row:
            return JSONResponse(status_code=404, content={"error": "Analysis not found."})
        return {"label": row["label"], "analysis": row["payload"]}
