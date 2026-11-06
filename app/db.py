"""
CommentPulse storage — users, social tokens, saved analyses.

Default: SQLite file (data/commentpulse.db). If DATABASE_URL is a Postgres
URL (e.g. a free Neon database), Postgres is used instead — recommended for
production because Render's free disk is ephemeral (SQLite data is lost on
redeploy/restart; logins and saved analyses would reset).

IDs are UUID hex strings generated here so both backends share one schema.
Access tokens are stored server-side only and never sent to the browser.
"""
import json
import os
import sqlite3
import time
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("CP_DB_PATH", BASE_DIR / "data" / "commentpulse.db"))
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
USE_PG = DATABASE_URL.startswith("postgres")

_pg = None
if USE_PG:
    try:
        import psycopg
        from psycopg.rows import dict_row
        _pg = (psycopg, dict_row)
    except Exception:
        USE_PG = False


def _connect():
    if USE_PG:
        psycopg, dict_row = _pg
        return psycopg.connect(DATABASE_URL, row_factory=dict_row)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _run(conn, sql, params=(), fetch=None):
    if USE_PG:
        sql = sql.replace("?", "%s")
        with conn.cursor() as cur:
            cur.execute(sql, params)
            if fetch == "one":
                row = cur.fetchone()
                conn.commit()
                return dict(row) if row else None
            if fetch == "all":
                rows = [dict(r) for r in cur.fetchall()]
                conn.commit()
                return rows
            conn.commit()
            return None
    cur = conn.execute(sql, params)
    if fetch == "one":
        row = cur.fetchone()
        conn.commit()
        return dict(row) if row else None
    if fetch == "all":
        rows = [dict(r) for r in cur.fetchall()]
        conn.commit()
        return rows
    conn.commit()
    return None


# ------------------------------------------------------------------ sessions
import hashlib
import secrets as _secrets


def _hash_sid(sid: str) -> str:
    return hashlib.sha256(sid.encode()).hexdigest()


def create_session(user_id: str, user_agent: str = "") -> str:
    """Create a server-side session row; returns the raw session id (cookie)."""
    sid = _secrets.token_urlsafe(32)
    now = time.time()
    conn = _connect()
    try:
        _run(conn, "INSERT INTO sessions (id_hash, user_id, created_at, expires_at, user_agent)"
                   " VALUES (?,?,?,?,?)",
             (_hash_sid(sid), user_id, now, now + 30 * 86400, (user_agent or "")[:200]))
        return sid
    finally:
        conn.close()


def get_session_user(sid: str):
    if not sid:
        return None
    conn = _connect()
    try:
        row = _run(conn, "SELECT * FROM sessions WHERE id_hash=?",
                   (_hash_sid(sid),), fetch="one")
        if not row or row["expires_at"] < time.time():
            return None
        return get_user(row["user_id"])
    finally:
        conn.close()


def delete_session(sid: str):
    if not sid:
        return
    conn = _connect()
    try:
        _run(conn, "DELETE FROM sessions WHERE id_hash=?", (_hash_sid(sid),))
    finally:
        conn.close()


def init_db():
    conn = _connect()
    try:
        _run(conn, """CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY, provider TEXT, provider_user_id TEXT,
            name TEXT, email TEXT, avatar TEXT, created_at REAL)""")
        _run(conn, """CREATE TABLE IF NOT EXISTS social_tokens (
            user_id TEXT, provider TEXT, access_token TEXT,
            expires_at REAL, updated_at REAL,
            PRIMARY KEY (user_id, provider))""")
        _run(conn, """CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY, user_id TEXT, label TEXT,
            payload TEXT, created_at REAL)""")
        _run(conn, """CREATE TABLE IF NOT EXISTS sessions (
            id_hash TEXT PRIMARY KEY, user_id TEXT,
            created_at REAL, expires_at REAL, user_agent TEXT)""")
    finally:
        conn.close()


def upsert_user(provider, provider_user_id, name, email, avatar):
    conn = _connect()
    try:
        row = _run(conn, "SELECT * FROM users WHERE provider=? AND provider_user_id=?",
                   (provider, provider_user_id), fetch="one")
        if row:
            _run(conn, "UPDATE users SET name=?, email=?, avatar=? WHERE id=?",
                 (name, email, avatar, row["id"]))
            return row["id"]
        uid = uuid.uuid4().hex
        _run(conn, "INSERT INTO users (id, provider, provider_user_id, name, email, avatar, created_at)"
                   " VALUES (?,?,?,?,?,?,?)",
             (uid, provider, provider_user_id, name, email, avatar, time.time()))
        return uid
    finally:
        conn.close()


def get_user(user_id):
    conn = _connect()
    try:
        return _run(conn, "SELECT * FROM users WHERE id=?", (user_id,), fetch="one")
    finally:
        conn.close()


def save_token(user_id, provider, access_token, expires_at=None):
    conn = _connect()
    try:
        existing = _run(conn, "SELECT user_id FROM social_tokens WHERE user_id=? AND provider=?",
                        (user_id, provider), fetch="one")
        if existing:
            _run(conn, "UPDATE social_tokens SET access_token=?, expires_at=?, updated_at=? WHERE user_id=? AND provider=?",
                 (access_token, expires_at, time.time(), user_id, provider))
        else:
            _run(conn, "INSERT INTO social_tokens (user_id, provider, access_token, expires_at, updated_at)"
                       " VALUES (?,?,?,?,?)",
                 (user_id, provider, access_token, expires_at, time.time()))
    finally:
        conn.close()


def get_token(user_id, provider):
    conn = _connect()
    try:
        row = _run(conn, "SELECT * FROM social_tokens WHERE user_id=? AND provider=?",
                   (user_id, provider), fetch="one")
        if not row:
            return None
        if row.get("expires_at") and row["expires_at"] < time.time():
            return None
        return row["access_token"]
    finally:
        conn.close()


def save_analysis(user_id, label, payload: dict) -> str:
    aid = uuid.uuid4().hex
    conn = _connect()
    try:
        _run(conn, "INSERT INTO analyses (id, user_id, label, payload, created_at) VALUES (?,?,?,?,?)",
             (aid, user_id, label, json.dumps(payload), time.time()))
        return aid
    finally:
        conn.close()


def list_analyses(user_id, limit=20):
    conn = _connect()
    try:
        rows = _run(conn, "SELECT id, label, created_at, payload FROM analyses WHERE user_id=? ORDER BY created_at DESC",
                    (user_id,), fetch="all") or []
        out = []
        for r in rows[:limit]:
            item = {"id": r["id"], "label": r["label"], "created_at": r["created_at"]}
            try:
                meta = (json.loads(r["payload"]) or {}).get("meta") or {}
                item["total_comments"] = meta.get("total_comments")
                item["avg_sentiment"] = meta.get("avg_sentiment")
            except Exception:
                pass
            out.append(item)
        return out
    finally:
        conn.close()


def get_analysis(user_id, analysis_id):
    conn = _connect()
    try:
        row = _run(conn, "SELECT * FROM analyses WHERE user_id=? AND id=?",
                   (user_id, analysis_id), fetch="one")
        if not row:
            return None
        row["payload"] = json.loads(row["payload"])
        return row
    finally:
        conn.close()


init_db()
