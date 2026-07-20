from __future__ import annotations

import os
import hmac as _hmac
import hashlib
import json
import time
import base64
import re
from pathlib import Path
import psycopg2
import psycopg2.extras
import bcrypt
from typing import Optional
from psycopg2.pool import ThreadedConnectionPool
from dotenv import load_dotenv

load_dotenv()

# Harden session secret validation in production
_ENV = os.environ.get("ENVIRONMENT", "development")
_SESSION_SECRET = os.environ.get("SESSION_SECRET")
if not _SESSION_SECRET:
    if _ENV.lower() == "production":
        raise RuntimeError("CRITICAL SECURITY ERROR: SESSION_SECRET env var must be set in a production environment!")
    else:
        _SESSION_SECRET = "medha-dev-secret-CHANGE-IN-PROD"

_pool = None

class PooledConnectionWrapper:
    """Wrapper that delegates all attributes/methods to the real connection
    but overrides close() to return it to the ThreadedConnectionPool instead of closing it."""
    def __init__(self, pool: ThreadedConnectionPool, conn):
        self._pool = pool
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        return self._conn.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._conn.__exit__(exc_type, exc_val, exc_tb)

    def close(self):
        # Return connection back to the pool
        self._pool.putconn(self._conn)

def get_conn():
    global _pool
    url = os.environ.get("DB_URL")
    if not url:
        raise RuntimeError("DB_URL is not set in .env")
    if _pool is None:
        # Scale to max 20 concurrent connections
        _pool = ThreadedConnectionPool(1, 20, url, cursor_factory=psycopg2.extras.RealDictCursor)
    conn = _pool.getconn()
    return PooledConnectionWrapper(_pool, conn)


# ──────────────────────────────────────────────
# DB Init
# ──────────────────────────────────────────────

def init_db():
    """Create tables if they don't exist. Call once on app startup."""
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id         SERIAL PRIMARY KEY,
                        name       VARCHAR(100)        NOT NULL,
                        email      VARCHAR(255) UNIQUE NOT NULL,
                        password   TEXT                NOT NULL,
                        created_at TIMESTAMP DEFAULT NOW()
                    );

                    CREATE TABLE IF NOT EXISTS blogs (
                        id         SERIAL PRIMARY KEY,
                        user_id    INTEGER REFERENCES users(id) ON DELETE CASCADE,
                        title      VARCHAR(500) NOT NULL,
                        slug       VARCHAR(500) NOT NULL,
                        content    TEXT         NOT NULL,
                        created_at TIMESTAMP DEFAULT NOW()
                    );

                    CREATE TABLE IF NOT EXISTS blog_images (
                        id         SERIAL PRIMARY KEY,
                        blog_id    INTEGER REFERENCES blogs(id) ON DELETE CASCADE,
                        filename   VARCHAR(255) NOT NULL,
                        image_data BYTEA        NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_blogs_user_id ON blogs(user_id);
                """)
    finally:
        conn.close()


# ──────────────────────────────────────────────
# Auth
# ──────────────────────────────────────────────

def is_valid_email(email: str) -> bool:
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email))

def register_user(name: str, email: str, password: str) -> "dict | str":
    """Returns user dict on success, error string on failure."""
    email = email.strip().lower()
    if not name.strip() or not email or not password:
        return "All fields are required."
    if not is_valid_email(email):
        return "Invalid email format."
    if len(password) < 6:
        return "Password must be at least 6 characters."

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (name, email, password) VALUES (%s, %s, %s) RETURNING id, name, email",
                    (name.strip(), email, hashed),
                )
                return dict(cur.fetchone())
    except psycopg2.errors.UniqueViolation:
        return "An account with this email already exists."
    except Exception as e:
        return f"Registration failed: {e}"
    finally:
        conn.close()


def login_user(email: str, password: str) -> "dict | None":
    """Returns user dict if credentials match, None otherwise."""
    email = email.strip().lower()
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, name, email, password FROM users WHERE email = %s",
                (email,),
            )
            row = cur.fetchone()
            if row and bcrypt.checkpw(password.encode(), row["password"].encode()):
                return {"id": row["id"], "name": row["name"], "email": row["email"]}
            return None
    finally:
        conn.close()


# ──────────────────────────────────────────────
# Blog Storage
# ──────────────────────────────────────────────

def save_blog(user_id: int, title: str, slug: str, content: str, images: list) -> int:
    """
    Save blog + images to DB. Returns blog_id.
    images: list of {"filename": str, "data": bytes}
    """
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO blogs (user_id, title, slug, content) VALUES (%s, %s, %s, %s) RETURNING id",
                    (user_id, title, slug, content),
                )
                blog_id = cur.fetchone()["id"]
                for img in images:
                    cur.execute(
                        "INSERT INTO blog_images (blog_id, filename, image_data) VALUES (%s, %s, %s)",
                        (blog_id, img["filename"], psycopg2.Binary(img["data"])),
                    )
                return blog_id
    finally:
        conn.close()


def get_user_blogs(user_id: int) -> list:
    """Returns all blogs for a user (id, title, slug, created_at), newest first."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, title, slug, created_at FROM blogs WHERE user_id = %s ORDER BY created_at DESC",
                (user_id,),
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_blog_content(blog_id: int, user_id: int) -> "dict | None":
    """Returns blog + images only if it belongs to user_id."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, title, slug, content FROM blogs WHERE id = %s AND user_id = %s",
                (blog_id, user_id),
            )
            row = cur.fetchone()
            if not row:
                return None
            blog = dict(row)
            cur.execute(
                "SELECT filename, image_data FROM blog_images WHERE blog_id = %s",
                (blog_id,),
            )
            blog["images"] = [
                {"filename": r["filename"], "data": bytes(r["image_data"])}
                for r in cur.fetchall()
            ]
            return blog
    finally:
        conn.close()


def delete_blog(blog_id: int, user_id: int) -> bool:
    """Delete a blog only if it belongs to user_id. Returns True on success."""
    conn = get_conn()
    try:
        # Query filenames associated with this blog so they can be deleted from disk
        with conn.cursor() as cur:
            cur.execute("SELECT filename FROM blog_images WHERE blog_id = %s", (blog_id,))
            filenames = [r["filename"] for r in cur.fetchall()]

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM blogs WHERE id = %s AND user_id = %s",
                    (blog_id, user_id),
                )
                deleted = cur.rowcount > 0

        if deleted and filenames:
            images_dir = Path("images")
            for fname in filenames:
                try:
                    file_path = images_dir / fname
                    if file_path.exists():
                        file_path.unlink()
                except Exception:
                    pass

        return deleted
    finally:
        conn.close()


# ──────────────────────────────────────────────
# Profile & Account Management
# ──────────────────────────────────────────────

def get_user_stats(user_id: int) -> dict:
    """Returns aggregate profile stats for the user in a single optimized query."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    u.name, 
                    u.email, 
                    u.created_at AS member_since,
                    COALESCE(b_stats.blog_count, 0) AS blog_count,
                    COALESCE(img_stats.image_count, 0) AS image_count,
                    b_stats.first_blog,
                    b_stats.last_blog
                FROM users u
                LEFT JOIN (
                    SELECT 
                        user_id, 
                        COUNT(*) AS blog_count, 
                        MIN(created_at) AS first_blog, 
                        MAX(created_at) AS last_blog
                    FROM blogs 
                    WHERE user_id = %s 
                    GROUP BY user_id
                ) b_stats ON u.id = b_stats.user_id
                LEFT JOIN (
                    SELECT 
                        b.user_id, 
                        COUNT(bi.id) AS image_count
                    FROM blog_images bi
                    JOIN blogs b ON bi.blog_id = b.id
                    WHERE b.user_id = %s
                    GROUP BY b.user_id
                ) img_stats ON u.id = img_stats.user_id
                WHERE u.id = %s;
            """, (user_id, user_id, user_id))
            row = cur.fetchone()
            if not row:
                raise ValueError("User not found")
            
            user_row = dict(row)
            return {
                "name": user_row["name"],
                "email": user_row["email"],
                "member_since": user_row["member_since"],
                "blog_count": int(user_row["blog_count"]),
                "image_count": int(user_row["image_count"]),
                "first_blog": user_row["first_blog"],
                "last_blog": user_row["last_blog"],
            }
    finally:
        conn.close()


def get_user_blogs_detail(user_id: int) -> list:
    """Returns blogs with estimated word count for the history view."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT b.id, b.title, b.slug, b.created_at,
                       LENGTH(b.content) AS char_count,
                       COUNT(bi.id) AS image_count
                FROM blogs b
                LEFT JOIN blog_images bi ON bi.blog_id = b.id
                WHERE b.user_id = %s
                GROUP BY b.id
                ORDER BY b.created_at DESC
            """, (user_id,))
            rows = []
            for r in cur.fetchall():
                d = dict(r)
                d["word_count"] = max(1, d["char_count"] // 5)  # rough estimate
                rows.append(d)
            return rows
    finally:
        conn.close()


def update_user_name(user_id: int, new_name: str) -> bool:
    """Update display name. Returns True on success."""
    if not new_name.strip():
        return False
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE users SET name = %s WHERE id = %s", (new_name.strip(), user_id))
                return cur.rowcount > 0
    finally:
        conn.close()


def change_password(user_id: int, old_password: str, new_password: str) -> str:
    """Returns 'ok' on success, error message on failure."""
    if len(new_password) < 6:
        return "New password must be at least 6 characters."
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT password FROM users WHERE id = %s", (user_id,))
            row = cur.fetchone()
            if not row or not bcrypt.checkpw(old_password.encode(), row["password"].encode()):
                return "Current password is incorrect."
        new_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
        with conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE users SET password = %s WHERE id = %s", (new_hash, user_id))
        return "ok"
    finally:
        conn.close()


def delete_user_account(user_id: int, password: str) -> str:
    """Deletes account + all data after verifying password. Returns 'ok' or error."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT password FROM users WHERE id = %s", (user_id,))
            row = cur.fetchone()
            if not row or not bcrypt.checkpw(password.encode(), row["password"].encode()):
                return "Password is incorrect."

        # Query all image filenames belonging to this user before deletion
        with conn.cursor() as cur:
            cur.execute("""
                SELECT bi.filename 
                FROM blog_images bi
                JOIN blogs b ON bi.blog_id = b.id
                WHERE b.user_id = %s
            """, (user_id,))
            filenames = [r["filename"] for r in cur.fetchall()]

        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM users WHERE id = %s", (user_id,))

        # Clean up files from disk
        if filenames:
            images_dir = Path("images")
            for fname in filenames:
                try:
                    file_path = images_dir / fname
                    if file_path.exists():
                        file_path.unlink()
                except Exception:
                    pass

        return "ok"
    finally:
        conn.close()


# ──────────────────────────────────────────────
# Session Tokens (for persistent cookie login)
# ──────────────────────────────────────────────

def create_session_token(user: dict) -> str:
    """
    Create a signed, base64-encoded session token valid for 30 days.
    Stored as a browser cookie so users stay logged in across refreshes.
    """
    payload = json.dumps({
        "id":    user["id"],
        "name":  user["name"],
        "email": user["email"],
        "exp":   int(time.time()) + 30 * 24 * 3600,  # 30 days
    }, sort_keys=True)
    sig = _hmac.new(_SESSION_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    raw = f"{payload}||{sig}"
    return base64.urlsafe_b64encode(raw.encode()).decode()


def verify_session_token(token: str) -> "dict | None":
    """
    Verify a session token. Returns user dict {id, name, email} if valid,
    None if tampered, expired, or malformed.
    """
    try:
        raw = base64.urlsafe_b64decode(token.encode()).decode()
        payload_str, sig = raw.split("||", 1)
        expected = _hmac.new(_SESSION_SECRET.encode(), payload_str.encode(), hashlib.sha256).hexdigest()
        if not _hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(payload_str)
        if payload.get("exp", 0) < int(time.time()):
            return None  # expired
        return {"id": payload["id"], "name": payload["name"], "email": payload["email"]}
    except Exception:
        return None


