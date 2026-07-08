from __future__ import annotations

import json
import os
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, List, Iterator, Tuple

import pandas as pd
import streamlit as st

# -----------------------------
# Import your compiled LangGraph app
# -----------------------------
from backend import app
try:
    from auth import (
        init_db, register_user, login_user,
        get_user_blogs, get_blog_content, delete_blog,
        get_user_stats, get_user_blogs_detail,
        update_user_name, change_password, delete_user_account,
        create_session_token, verify_session_token,
    )
    _AUTH_AVAILABLE = True
except ImportError:
    _AUTH_AVAILABLE = False

try:
    import extra_streamlit_components as stx
    _COOKIES_AVAILABLE = True
except ImportError:
    _COOKIES_AVAILABLE = False


# -----------------------------
# Helpers
# -----------------------------
def show_loading_screen(message: str, subtitle: str):
    """Render a premium liquid glass loading/teleporting overlay."""
    st.markdown(f"""
    <div style="display:flex; justify-content:center; align-items:center; height:80vh; font-family:'Courier',monospace; color:#cbd5e1;">
        <div style="text-align:center; background:rgba(255,255,255,0.02); padding:2.5rem; border:1px solid rgba(255,255,255,0.08); border-radius:20px; backdrop-filter:blur(32px); box-shadow:0 8px 32px rgba(0,0,0,0.37);">
            <div style="font-size:1.8rem; font-weight:700; margin-bottom:0.8rem; letter-spacing:3px; background:linear-gradient(135deg,#a5b4fc 0%,#2dd4bf 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; animation:pulse 1.2s infinite;">{message}</div>
            <div style="font-size:0.8rem; color:#64748b; text-transform:uppercase; letter-spacing:2px;">{subtitle}</div>
        </div>
    </div>
    <style>
    @keyframes pulse {{
        0%, 100% {{ opacity: 0.6; }}
        50% {{ opacity: 1; }}
    }}
    body {{
        background-color: #0b0d19 !important;
    }}
    </style>
    """, unsafe_allow_html=True)


def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))

        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def images_zip(images_dir: Path) -> Optional[bytes]:
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p))
    return buf.getvalue()


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Stream graph progress if available; else invoke.
    Yields ("updates"/"values"/"final", payload).
    """
    try:
        for step in graph_app.stream(inputs, stream_mode="updates"):
            yield ("updates", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass

    try:
        for step in graph_app.stream(inputs, stream_mode="values"):
            yield ("values", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass

    out = graph_app.invoke(inputs)
    yield ("final", out)


def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            inner = next(iter(step_payload.values()))
            current_state.update(inner)
        else:
            current_state.update(step_payload)
    return current_state


# -----------------------------
# Markdown renderer that supports local images
# -----------------------------
_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def _resolve_image_path(src: str) -> Path:
    src = src.strip().lstrip("./")
    return Path(src).resolve()


def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last : m.start()]
        if before:
            parts.append(("md", before))

        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()

    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]

        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = payload.split("|||", 1)

        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    rest = "\n".join(nxt.splitlines()[1:])
                    parts[i + 1] = ("md", rest)

        if src.startswith("http://") or src.startswith("https://"):
            st.image(src, caption=caption or (alt or None), use_container_width=True)
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists():
                st.image(str(img_path), caption=caption or (alt or None), use_container_width=True)
            else:
                st.warning(f"Image not found: `{src}` (looked for `{img_path}`)")

        i += 1


# -----------------------------
# ✅ NEW: Past blogs helpers
# -----------------------------
def list_past_blogs() -> List[Path]:
    """
    Returns .md files in current working directory, newest first.
    Filters out obvious non-blog markdown files if needed.
    """
    cwd = Path(".")
    files = [p for p in cwd.glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    """
    Use first '# ' heading as title if present.
    """
    for line in md.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="InkGraph",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialise DB tables (no-op if DB not configured)
if _AUTH_AVAILABLE:
    try:
        init_db()
    except Exception:
        pass

# ── Cookie manager (must be initialised before any st.stop()) ────
_cookie_manager = stx.CookieManager(key="inkgraph_cm") if _COOKIES_AVAILABLE else None

# Handle pending cookie deletion on logout
if _COOKIES_AVAILABLE and _cookie_manager and st.session_state.get("logout_pending"):
    st.session_state.pop("logout_pending", None)
    try:
        _cookie_manager.delete("inkgraph_session", key="logout_delete_cookie")
    except Exception:
        pass

# Auto-restore session from cookie on page load / refresh
if _AUTH_AVAILABLE and _COOKIES_AVAILABLE and "user" not in st.session_state:
    if not st.session_state.get("logged_out", False):
        if "cookie_checked" not in st.session_state:
            st.session_state["cookie_checked"] = False

        cookies = _cookie_manager.get_all()
        if not cookies and not st.session_state["cookie_checked"]:
            st.session_state["cookie_checked"] = True
            show_loading_screen("TELEPORTING...", "Securing Session Tunnel")
            st.stop()
        else:
            if cookies:
                _token = cookies.get("inkgraph_session")
                if _token:
                    _user_from_cookie = verify_session_token(str(_token))
                    if _user_from_cookie:
                        st.session_state["user"] = _user_from_cookie
                        st.session_state["page"] = "home"

# ── Auth helpers ─────────────────────────────────────────────
def _render_auth_page():
    """Full-page login / register UI shown when user is not logged in."""
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center; padding:2.5rem 0 1.5rem 0;">
            <!-- <div style="font-size:3.5rem; margin-bottom:0.5rem;">🖊️</div> -->
            <h1 style="font-family:'Courier',monospace; font-size:2.4rem; font-weight:800;
                background:linear-gradient(135deg,#a5b4fc 0%,#818cf8 50%,#2dd4bf 100%);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                background-clip:text; margin:0 0 0.3rem 0;">InkGraph</h1>
            <p style="color:#94a3b8; font-family:'Courier',monospace; font-size:0.9rem;
                letter-spacing:1px; margin:0;">NEURAL BLOG GENERATING AGENT</p>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_reg = st.tabs(["🔑  Login", "📝  Register"])

        with tab_login:
            if "login_error" in st.session_state:
                st.error(st.session_state.pop("login_error"))
            with st.form("login_form", clear_on_submit=False):
                email_in = st.text_input("Email", placeholder="you@example.com")
                pass_in  = st.text_input("Password", type="password", placeholder="********")
                if st.form_submit_button("Login", use_container_width=True, type="primary"):
                    if not email_in or not pass_in:
                        st.error("Please fill in all fields.")
                    else:
                        st.session_state["login_pending"] = {"email": email_in, "password": pass_in}
                        st.rerun()

        with tab_reg:
            if "register_error" in st.session_state:
                st.error(st.session_state.pop("register_error"))
            with st.form("register_form", clear_on_submit=False):
                name_in    = st.text_input("Full Name", placeholder="John Doe")
                email_in2  = st.text_input("Email", placeholder="you@example.com", key="reg_email")
                pass_in2   = st.text_input("Password", type="password",
                                           placeholder="Min 6 characters", key="reg_pass")
                confirm_in = st.text_input("Confirm Password", type="password",
                                           placeholder="Repeat password", key="reg_confirm")
                if st.form_submit_button("Create Account", use_container_width=True, type="primary"):
                    if pass_in2 != confirm_in:
                        st.error("Passwords do not match.")
                    elif not name_in.strip() or not email_in2.strip() or not pass_in2:
                        st.error("All fields are required.")
                    else:
                        st.session_state["register_pending"] = {
                            "name": name_in,
                            "email": email_in2,
                            "password": pass_in2
                        }
                        st.rerun()


def _render_profile_page(user: dict):
    """Premium liquid glass user profile page."""

    uid = user["id"]
    stats = st.session_state.get("profile_stats")
    blogs = st.session_state.get("profile_blogs")
    if stats is None or blogs is None:
        stats = get_user_stats(uid)
        blogs = get_user_blogs_detail(uid)

    initials = "".join(p[0].upper() for p in stats["name"].split()[:2])
    member_since = stats["member_since"]
    ms_str = member_since.strftime("%b %Y") if hasattr(member_since, "strftime") else str(member_since)[:7]

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    days_active = (now - member_since.replace(tzinfo=timezone.utc)).days if member_since else 0

    # ── Liquid glass CSS (profile-specific) ──────────────────
    st.markdown("""
    <style>
    .pg-hero {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(32px) saturate(200%);
        -webkit-backdrop-filter: blur(32px) saturate(200%);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 28px;
        padding: 2.5rem 2rem;
        display: flex;
        align-items: center;
        gap: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 40px rgba(0,0,0,0.35),
                    inset 0 1px 0 rgba(255,255,255,0.12),
                    inset 0 -1px 0 rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
    }
    .pg-hero::before {
        content: '';
        position: absolute; inset: 0;
        background: linear-gradient(135deg,
            rgba(165,180,252,0.08) 0%,
            rgba(45,212,191,0.05) 50%,
            rgba(139,92,246,0.08) 100%);
        pointer-events: none;
    }
    .pg-hero::after {
        content: '';
        position: absolute;
        top: -60%; left: -30%;
        width: 120%; height: 120%;
        background: radial-gradient(ellipse, rgba(165,180,252,0.08) 0%, transparent 70%);
        animation: liquidShimmer 6s ease-in-out infinite alternate;
        pointer-events: none;
    }
    @keyframes liquidShimmer {
        0%   { transform: translate(0,0) scale(1); }
        100% { transform: translate(10%,5%) scale(1.05); }
    }
    .pg-avatar {
        width: 90px; height: 90px; min-width: 90px;
        border-radius: 50%;
        background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #2dd4bf 100%);
        display: flex; align-items: center; justify-content: center;
        font-family: 'Courier Prime', 'Courier', monospace;
        font-size: 2rem; font-weight: 800; color: #fff;
        box-shadow: 0 0 28px rgba(165,180,252,0.45), 0 0 56px rgba(45,212,191,0.2);
        animation: avatarPulse 3.5s ease-in-out infinite;
        position: relative; z-index: 1;
    }
    @keyframes avatarPulse {
        0%,100% { box-shadow: 0 0 28px rgba(165,180,252,0.45), 0 0 56px rgba(45,212,191,0.2); }
        50%     { box-shadow: 0 0 40px rgba(165,180,252,0.7), 0 0 70px rgba(45,212,191,0.35); }
    }
    .pg-name  { font-family:'Courier Prime', 'Courier', monospace; font-size:1.7rem; font-weight:800;
                color:#f1f5f9; margin:0 0 4px 0; letter-spacing:-0.3px; }
    .pg-email { font-family:'Courier Prime', 'Courier', monospace; font-size:0.88rem; color:#64748b; margin:0 0 10px 0; }
    .pg-badge {
        display:inline-block; padding:4px 12px;
        background:rgba(99,102,241,0.15); border:1px solid rgba(99,102,241,0.3);
        border-radius:20px; font-family:'Courier Prime', 'Courier', monospace;
        font-size:0.75rem; font-weight:600; color:#a5b4fc; letter-spacing:0.5px;
    }
    .stat-card {
        background: rgba(255,255,255,0.035);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 18px;
        padding: 1.4rem 1.2rem;
        text-align: center;
        position: relative; overflow: hidden;
        box-shadow: 0 4px 24px rgba(0,0,0,0.25),
                    inset 0 1px 0 rgba(255,255,255,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 36px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.14);
    }
    .stat-card::before {
        content:''; position:absolute; inset:0;
        background:linear-gradient(135deg,rgba(255,255,255,0.03) 0%,transparent 60%);
        pointer-events:none;
    }
    .stat-num   { font-family:'Courier Prime', 'Courier', monospace; font-size:2.2rem; font-weight:800;
                  color:#f1f5f9; margin:0 0 4px 0; }
    .stat-label { font-family:'Courier Prime', 'Courier', monospace; font-size:0.78rem; color:#64748b;
                  text-transform:uppercase; letter-spacing:1px; }
    .stat-icon  { font-size:1.6rem; margin-bottom:0.5rem; display:block; }
    
    /* Native Streamlit border container overrides for Glassmorphism */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,0.03) !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 20px !important;
        padding: 1.6rem !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 4px 32px rgba(0,0,0,0.2),
                    inset 0 1px 0 rgba(255,255,255,0.06) !important;
    }
    
    /* Special tint for Danger Zone block */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.danger-header) {
        background: rgba(239,68,68,0.05) !important;
        border: 1px solid rgba(239,68,68,0.2) !important;
        box-shadow: 0 4px 32px rgba(239,68,68,0.1),
                    inset 0 1px 0 rgba(255,255,255,0.02) !important;
    }

    .blog-row {
        display:flex; justify-content:space-between; align-items:center;
        padding:0.85rem 1rem;
        background:rgba(255,255,255,0.02);
        border:1px solid rgba(255,255,255,0.06);
        border-radius:12px;
        margin-bottom:0.6rem;
        transition: background 0.2s ease, border-color 0.2s ease;
    }
    .blog-row:hover { background:rgba(165,180,252,0.06); border-color:rgba(165,180,252,0.2); }
    .blog-title { font-family:'Courier Prime', 'Courier', monospace; font-size:0.95rem;
                  font-weight:600; color:#e2e8f0; }
    .blog-meta  { font-family:'Courier Prime', 'Courier', monospace; font-size:0.78rem; color:#64748b; margin-top:2px; }
    .pg-section-title {
        font-family:'Courier Prime', 'Courier', monospace; font-size:1.1rem; font-weight:700;
        color:#f1f5f9; margin:0 0 1.2rem 0; letter-spacing:0.3px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Back navigation ──────────────────────────────────────
    if st.button("← Back to InkGraph", key="back_from_profile"):
        st.session_state["back_to_home_pending"] = True
        st.rerun()

    # ── Hero profile card ─────────────────────────────────────
    st.markdown(f"""
    <div class="pg-hero">
        <div class="pg-avatar">{initials}</div>
        <div style="position:relative;z-index:1;">
            <div class="pg-name">{stats['name']}</div>
            <div class="pg-email">{stats['email']}</div>
            <span class="pg-badge">⭐ Member since {ms_str}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Stat cards ────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "📝", str(stats["blog_count"]), "Blogs Generated"),
        (c2, "🖼️", str(stats["image_count"]), "Images Created"),
        (c3, "📅", str(days_active), "Days Active"),
        (c4, "⚡", str(stats["blog_count"] * 6), "Sections Written"),
    ]
    for col, icon, num, label in cards:
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <span class="stat-icon">{icon}</span>
                <div class="stat-num">{num}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1.4, 1], gap="large")

    # ── Blog History ──────────────────────────────────────────
    with left_col:
        with st.container(border=True):
            st.markdown("""
            <div class="pg-section-title" style="margin-bottom:1rem;">Blog History</div>
            """, unsafe_allow_html=True)
            if not blogs:
                st.caption("No blogs generated yet. Go generate your first one!")
            else:
                for b in blogs:
                    dt = b["created_at"]
                    dt_str = dt.strftime("%d %b %Y") if hasattr(dt, "strftime") else str(dt)[:10]
                    img_str = f"{b['image_count']} img{'s' if b['image_count'] != 1 else ''}"
                    col_info, col_del = st.columns([5, 1])
                    with col_info:
                        st.markdown(f"""
                        <div class="blog-row">
                            <div class="blog-title">{b['title']}</div>
                            <div class="blog-meta">{dt_str} &nbsp;·&nbsp; ~{b['word_count']:,} words &nbsp;·&nbsp; {img_str}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_del:
                        if st.button("🗑", key=f"del_{b['id']}", help=f"Delete '{b['title'][:40]}'"):
                            if delete_blog(b["id"], uid):
                                st.session_state.pop("profile_stats", None)
                                st.session_state.pop("profile_blogs", None)
                                st.success("Deleted.")
                                st.rerun()

    # ── Account Settings ──────────────────────────────────────
    with right_col:
        with st.container(border=True):
            st.markdown("""
            <div class="pg-section-title" style="margin-bottom:1rem;">Account Settings</div>
            """, unsafe_allow_html=True)

            with st.expander("✏️ Change Display Name", expanded=False):
                with st.form("change_name_form"):
                    new_name = st.text_input("New Name", value=stats["name"])
                    if st.form_submit_button("Update Name", type="primary", use_container_width=True):
                        if update_user_name(uid, new_name):
                            st.session_state["user"]["name"] = new_name.strip()
                            st.session_state.pop("profile_stats", None)
                            st.success("Name updated!")
                            st.rerun()
                        else:
                            st.error("Name cannot be empty.")

            with st.expander("🔒 Change Password", expanded=False):
                with st.form("change_pass_form"):
                    old_p = st.text_input("Current Password", type="password")
                    new_p = st.text_input("New Password", type="password", placeholder="Min 6 characters")
                    cnf_p = st.text_input("Confirm New Password", type="password")
                    if st.form_submit_button("Update Password", type="primary", use_container_width=True):
                        if new_p != cnf_p:
                            st.error("Passwords do not match.")
                        else:
                            res = change_password(uid, old_p, new_p)
                            if res == "ok":
                                st.success("Password updated successfully!")
                            else:
                                st.error(res)

        # Danger zone
        with st.container(border=True):
            st.markdown("""
            <div class="danger-header pg-section-title" style="color:#f87171; margin-bottom:0.3rem;">⚠️ Danger Zone</div>
            <div style="font-family:'Courier Prime', 'Courier', monospace; font-size:0.8rem; color:#64748b; margin-bottom:0.8rem;">
                Permanently deletes your account and all generated blogs.
            </div>
            """, unsafe_allow_html=True)
            with st.expander("🗑 Delete My Account", expanded=False):
                with st.form("delete_acc_form"):
                    confirm_pass = st.text_input("Enter your password to confirm", type="password")
                    if st.form_submit_button("Delete Account Forever", use_container_width=True):
                        res = delete_user_account(uid, confirm_pass)
                        if res == "ok":
                            st.session_state.clear()
                            st.success("Account deleted.")
                            st.rerun()
                        else:
                            st.error(res)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Courier+Prime:ital,wght@0,400;0,700;1,400;1,700&family=Share+Tech+Mono&display=swap');

/* Base Override */
html, body, [class*="css"] {
    font-family: 'Courier Prime', 'Courier', monospace;
    color: #cbd5e1;
    scroll-behavior: smooth;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Courier Prime', 'Courier', monospace;
    font-weight: 700;
    letter-spacing: -0.5px;
}

.stApp {
    background-color: #080914 !important;
    background-image: 
        radial-gradient(at 0% 0%, rgba(129, 140, 248, 0.12) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(139, 92, 246, 0.1) 0px, transparent 50%),
        radial-gradient(at 50% 100%, rgba(45, 212, 191, 0.08) 0px, transparent 60%) !important;
    min-height: 100vh;
    position: relative;
    overflow-x: hidden;
}

/* Soothing floating color blobs */
.stApp::before {
    content: '';
    position: absolute;
    top: 10%;
    left: 15%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(129, 140, 248, 0.12) 0%, transparent 70%);
    filter: blur(100px);
    pointer-events: none;
    z-index: 0;
    animation: blobFloat 25s infinite ease-in-out alternate;
}

.stApp::after {
    content: '';
    position: absolute;
    bottom: 15%;
    right: 15%;
    width: 550px;
    height: 550px;
    background: radial-gradient(circle, rgba(45, 212, 191, 0.08) 0%, transparent 70%);
    filter: blur(100px);
    pointer-events: none;
    z-index: 0;
    animation: blobFloat 30s infinite ease-in-out alternate-reverse;
}

/* Glassmorphic Sidebar styling */
[data-testid="stSidebar"] {
    background: rgba(11, 13, 28, 0.6) !important;
    backdrop-filter: blur(35px) !important;
    -webkit-backdrop-filter: blur(35px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    box-shadow: 15px 0 45px rgba(0, 0, 0, 0.45) !important;
    z-index: 100;
}

/* Sidebar header */
.sidebar-header {
    font-family: 'Courier Prime', 'Courier', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: #f1f5f9;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    padding-bottom: 12px;
    margin-bottom: 22px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* Sidebar Widget labels */
[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p,
label[data-testid="stWidgetLabel"] p {
    color: #94a3b8 !important;
    font-family: 'Courier Prime', 'Courier', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px;
}

/* Text fields and Textareas - Frosted */
.stTextArea textarea, .stTextInput input, .stDateInput input {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
    font-family: 'Courier Prime', 'Courier', monospace !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2) !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    padding: 0.65rem 0.9rem !important;
}

.stTextArea textarea:focus, .stTextInput input:focus, .stDateInput input:focus {
    background: rgba(255, 255, 255, 0.04) !important;
    border-color: rgba(165, 180, 252, 0.5) !important; /* Soft Lavender */
    box-shadow: 0 0 20px rgba(165, 180, 252, 0.2) !important;
}

/* Primary generate button with soft gradient glow styling */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(129, 140, 248, 0.8) 0%, rgba(45, 212, 191, 0.8) 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    backdrop-filter: blur(12px) !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.5rem !important;
    font-family: 'Courier Prime', 'Courier', monospace !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
    box-shadow: 0 4px 20px rgba(129, 140, 248, 0.25) !important;
    cursor: pointer;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(129, 140, 248, 0.35), 0 0 20px rgba(45, 212, 191, 0.25) !important;
    border-color: rgba(255, 255, 255, 0.3) !important;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"]:active {
    transform: translateY(0px) !important;
}

/* Secondary Button inside Sidebar (e.g. Load selected blog) */
[data-testid="stSidebar"] .stButton > button:not([kind="primary"]) {
    background: rgba(255, 255, 255, 0.03) !important;
    color: #cbd5e1 !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important;
    font-family: 'Courier Prime', 'Courier', monospace !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    cursor: pointer;
}

[data-testid="stSidebar"] .stButton > button:not([kind="primary"]):hover {
    background: rgba(255, 255, 255, 0.08) !important;
    color: #ffffff !important;
    border-color: rgba(255, 255, 255, 0.15) !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    transform: translateY(-1px) !important;
}

/* Radio buttons list in sidebar */
[data-testid="stRadio"] label {
    font-family: 'Courier Prime', 'Courier', monospace !important;
    font-size: 0.9rem !important;
    color: #94a3b8 !important;
    transition: all 0.2s ease;
}

[data-testid="stRadio"] label:hover {
    color: #ffffff !important;
}

/* Tabs: Cyber Holographic styling */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255, 255, 255, 0.02) !important;
    border-radius: 16px !important;
    padding: 0.4rem !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    gap: 8px !important;
    margin-bottom: 2rem !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Courier Prime', 'Courier', monospace !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    padding: 0.5rem 1.2rem !important;
    border-radius: 12px !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    border: 1px solid transparent !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #ffffff !important;
    background: rgba(255, 255, 255, 0.04) !important;
}

.stTabs [aria-selected="true"] {
    background: rgba(129, 140, 248, 0.15) !important;
    color: #ffffff !important;
    border: 1px solid rgba(129, 140, 248, 0.3) !important;
    box-shadow: 0 4px 20px rgba(129, 140, 248, 0.2) !important;
}

/* Frosted Glass Panels, Dataframes & Expanders */
.stDataFrame, [data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(25px) !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

[data-testid="stExpander"]:hover {
    background: rgba(255, 255, 255, 0.04) !important;
    border-color: rgba(129, 140, 248, 0.2) !important;
    box-shadow: 0 8px 32px 0 rgba(129, 140, 248, 0.1) !important;
}

/* Status box override */
[data-testid="stStatus"] {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 14px !important;
    color: #cbd5e1 !important;
    box-shadow: 0 4px 25px rgba(0,0,0,0.2) !important;
}

/* Download buttons */
.stDownloadButton > button {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important;
    color: #cbd5e1 !important;
    font-family: 'Courier Prime', 'Courier', monospace !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    padding: 0.65rem 1.4rem !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.stDownloadButton > button:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    color: #ffffff !important;
    border-color: rgba(255, 255, 255, 0.15) !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    transform: translateY(-1px) !important;
}

/* Custom Scrollbars */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.1); }
::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.1); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.2); }

/* Custom Animations & Component Styles */
.cyber-header {
    position: relative;
    padding: 2.5rem 1.5rem;
    margin-bottom: 2.2rem;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 10px 40px rgba(0,0,0,0.35), inset 0 0 20px rgba(255,255,255,0.01);
    backdrop-filter: blur(30px);
    overflow: hidden;
    animation: slideUp 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}

.cyber-scanline {
    position: absolute;
    top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.03) 0%, transparent 60%);
    animation: rotateLiquid 25s linear infinite;
    pointer-events: none;
}

.cyber-title {
    font-family: 'Courier Prime', 'Courier', monospace !important;
    font-size: 3rem !important;
    font-weight: 800 !important;
    margin: 0 0 0.5rem 0 !important;
    background: linear-gradient(135deg, #a5b4fc 0%, #818cf8 50%, #2dd4bf 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    text-shadow: 0 2px 15px rgba(0, 0, 0, 0.3);
    letter-spacing: -1px !important;
}

.cyber-subtitle {
    font-family: 'Courier Prime', 'Courier', monospace;
    font-weight: 400;
    font-size: 0.95rem;
    color: #94a3b8;
    margin-bottom: 1rem;
    letter-spacing: 0.5px;
}

.cyber-badge-container {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin-top: 12px;
}

.cyber-badge {
    display: inline-block;
    padding: 0.35rem 0.9rem;
    font-family: 'Courier Prime', 'Courier', monospace;
    font-weight: 500;
    font-size: 0.75rem;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    color: #cbd5e1;
    border-radius: 8px;
    letter-spacing: 0.5px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
}

.cyber-ready-card {
    background: rgba(255, 255, 255, 0.01);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 28px;
    padding: 3.5rem 2rem;
    text-align: center;
    max-width: 650px;
    margin: 3.5rem auto;
    box-shadow: 0 15px 45px rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(25px);
    animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}

.cyber-ready-icon {
    font-size: 4rem;
    margin-bottom: 1.5rem;
    filter: drop-shadow(0 4px 15px rgba(255, 255, 255, 0.15));
    animation: float 4s ease-in-out infinite;
}

.cyber-ready-title {
    font-family: 'Courier Prime', 'Courier', monospace;
    color: #ffffff;
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.75rem;
    letter-spacing: -0.5px;
}

/* Glassmorphic border containers styling (global) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 20px !important;
    padding: 1.6rem !important;
    margin-bottom: 1.5rem !important;
    box-shadow: 0 4px 32px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.06) !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.danger-header) {
    background: rgba(239, 68, 68, 0.05) !important;
    border: 1px solid rgba(239, 68, 68, 0.2) !important;
    box-shadow: 0 4px 32px rgba(239, 68, 68, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.02) !important;
}

/* Animations */
@keyframes rotateLiquid {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-8px); }
    100% { transform: translateY(0px); }
}

@keyframes slideUp {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}

@keyframes blobFloat {
    0% { transform: translate(0, 0) scale(1); }
    50% { transform: translate(40px, -30px) scale(1.08); }
    100% { transform: translate(-30px, 40px) scale(0.95); }
}
</style>
""", unsafe_allow_html=True)

# Intercept pending auth database load actions (shows loading screen while querying)
if "login_pending" in st.session_state:
    show_loading_screen("TELEPORTING...", "Verifying Credentials")
    pending = st.session_state.pop("login_pending")
    user = login_user(pending["email"], pending["password"])
    if user:
        st.session_state["user"] = user
        st.session_state["logged_out"] = False
        if _COOKIES_AVAILABLE and _cookie_manager:
            _cookie_manager.set(
                "inkgraph_session",
                create_session_token(user),
                max_age=30 * 24 * 3600,
                key="login_cookie_set"
            )
        st.session_state["page"] = "home"
    else:
        st.session_state["login_error"] = "Invalid email or password."
    st.rerun()

if "register_pending" in st.session_state:
    show_loading_screen("TELEPORTING...", "Creating Secure Account")
    pending = st.session_state.pop("register_pending")
    result = register_user(pending["name"], pending["email"], pending["password"])
    if isinstance(result, dict):
        st.session_state["user"] = result
        st.session_state["logged_out"] = False
        if _COOKIES_AVAILABLE and _cookie_manager:
            _cookie_manager.set(
                "inkgraph_session",
                create_session_token(result),
                max_age=30 * 24 * 3600,
                key="register_cookie_set"
            )
        st.session_state["page"] = "home"
    else:
        st.session_state["register_error"] = str(result)
    st.rerun()

# ── Auth gate ────────────────────────────────────────────────
if _AUTH_AVAILABLE and "user" not in st.session_state:
    _render_auth_page()
    st.stop()

_current_user: Dict[str, Any] = st.session_state.get("user", {})

# Intercept pending profile data load actions
if "profile_pending" in st.session_state:
    show_loading_screen("TELEPORTING...", "Loading Profile Matrix")
    st.session_state.pop("profile_pending")
    try:
        uid = _current_user["id"]
        st.session_state["profile_stats"] = get_user_stats(uid)
        st.session_state["profile_blogs"] = get_user_blogs_detail(uid)
    except Exception:
        pass
    st.session_state["page"] = "profile"
    st.rerun()

if "back_to_home_pending" in st.session_state:
    show_loading_screen("TELEPORTING...", "Loading Dashboard Matrix")
    st.session_state.pop("back_to_home_pending")
    st.session_state["page"] = "home"
    st.rerun()

# ── Page routing ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state["page"] = "home"

if st.session_state["page"] == "profile" and _AUTH_AVAILABLE and _current_user:
    _render_profile_page(_current_user)
    st.stop()

# ── Hero header ──────────────────────────────────────────────
st.markdown("""
<div class="cyber-header">
  <div class="cyber-scanline"></div>
  <h1 class="cyber-title">InkGraph</h1>
  <div class="cyber-subtitle">NEURAL BLOG GENERATING AGENT | POWERED BY LANGGRAPH</div>
  <div class="cyber-badge-container">
    <span class="cyber-badge">SYSTEM STATUS: ONLINE</span>
    <span class="cyber-badge" style="border-color: rgba(255,255,255,0.15); color: #cbd5e1;">CORE: ACTIVE</span>
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    # ── User info + logout ────────────────────────────────
    if _current_user:
        # st.markdown(f"""
        # <div style="padding-bottom:12px; border-bottom:1px solid rgba(255,255,255,0.08);
        #     margin-bottom:16px;">
        #     <div style="font-family:'Courier',monospace; font-size:0.78rem;
        #         color:#94a3b8; margin-bottom:2px;">Logged in as</div>
        #     <div style="font-family:'Courier',monospace; font-size:1rem;
        #         font-weight:700; color:#f1f5f9;">👤 {_current_user['name']}</div>
        #     <div style="font-family:'Courier',monospace; font-size:0.78rem;
        #         color:#64748b;">{_current_user['email']}</div>
        # </div>
        # """, unsafe_allow_html=True)
        if st.button("👤 My Profile", use_container_width=True):
            st.session_state["profile_pending"] = True
            st.rerun()
        if st.button("Logout", use_container_width=True):
            st.session_state["logout_pending"] = True
            st.session_state["logged_out"] = True
            st.session_state.pop("user", None)
            st.session_state.pop("last_out", None)
            st.session_state.pop("profile_stats", None)
            st.session_state.pop("profile_blogs", None)
            st.session_state["cookie_checked"] = True
            st.rerun()

    # ── Generate section ─────────────────────────────────
    st.markdown("""
    <div class="sidebar-header">
      ⚡ Generate New Blog
    </div>
    """, unsafe_allow_html=True)

    topic = st.text_area(
        "📌 Topic",
        height=120,
        placeholder="e.g. How Transformer attention works…",
    )
    as_of = st.date_input("📅 As of date", value=date.today())
    run_btn = st.button("Generate Blog", type="primary")

    st.divider()
    st.markdown("""
    <div style="font-family:'Courier',monospace; font-size:0.9rem; font-weight:600;
        color:#e2e8f0; margin-bottom:0.5rem; letter-spacing:0.5px; text-transform:uppercase;">
        📚 My Blogs
    </div>""", unsafe_allow_html=True)

    if _AUTH_AVAILABLE and _current_user:
        _user_blogs = get_user_blogs(_current_user["id"])
    else:
        _user_blogs = []

    if not _user_blogs:
        st.caption("No blogs yet — generate your first one!")
        selected_blog_id: Optional[int] = None
    else:
        _blog_opts: List[str] = []
        _blog_id_map: Dict[str, int] = {}
        for b in _user_blogs[:50]:
            dt = b["created_at"]
            dt_str = dt.strftime("%b %d, %Y") if hasattr(dt, "strftime") else str(dt)[:10]
            label = f"{b['title']}  ·  {dt_str}"
            _blog_opts.append(label)
            _blog_id_map[label] = b["id"]

        _sel_label = st.radio(
            "Select a blog to load",
            options=_blog_opts,
            index=0,
            label_visibility="collapsed",
        )
        selected_blog_id = _blog_id_map.get(_sel_label)

        if st.button("Load selected blog"):
            if selected_blog_id and _current_user:
                _blog = get_blog_content(selected_blog_id, _current_user["id"])
                if _blog:
                    _images_dir = Path("images")
                    _images_dir.mkdir(exist_ok=True)
                    for _img in _blog.get("images", []):
                        (_images_dir / _img["filename"]).write_bytes(_img["data"])
                    st.session_state["last_out"] = {
                        "plan": None,
                        "evidence": [],
                        "image_specs": [],
                        "final": _blog["content"],
                    }
                    st.success("Blog loaded.")
                    st.rerun()

    st.divider()
    st.markdown("© 2026 Raman - All rights reserved.")
    

# Keep your topic input as-is; optionally prefill for next run after loading a blog
if "topic_prefill" in st.session_state and isinstance(st.session_state["topic_prefill"], str):
    # Do not mutate widgets; just keep as a hint.
    pass

# Storage for latest run
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

# Layout
tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs(
    ["🧩 Plan", "🔎 Evidence", "📝 Markdown Preview", "🖼️ Images", "🧾 Logs"]
)

logs: List[str] = []


def log(msg: str):
    logs.append(msg)


if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    inputs: Dict[str, Any] = {
        "topic": topic.strip(),
        "user_id": _current_user.get("id") if _current_user else None,
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of.isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }

    status = st.status("Running graph…", expanded=True)
    progress_area = st.empty()

    current_state: Dict[str, Any] = {}
    last_node = None

    for kind, payload in try_stream(app, inputs):
        if kind in ("updates", "values"):
            node_name = None
            if isinstance(payload, dict) and len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
                node_name = next(iter(payload.keys()))
            if node_name and node_name != last_node:
                status.write(f"➡️ Node: `{node_name}`")
                last_node = node_name

            current_state = extract_latest_state(current_state, payload)

            summary = {
                "mode": current_state.get("mode"),
                "needs_research": current_state.get("needs_research"),
                "queries": current_state.get("queries", [])[:5] if isinstance(current_state.get("queries"), list) else [],
                "evidence_count": len(current_state.get("evidence", []) or []),
                "tasks": len((current_state.get("plan") or {}).get("tasks", [])) if isinstance(current_state.get("plan"), dict) else None,
                "images": len(current_state.get("image_specs", []) or []),
                "sections_done": len(current_state.get("sections", []) or []),
            }
            
            # Cybernetic dynamic telemetry dashboard
            mode_val = str(summary.get("mode") or "INITIALIZING").upper()
            needs_research_val = "YES" if summary.get("needs_research") else "NO"
            queries_count = len(summary.get("queries") or [])
            evidence_count = summary.get("evidence_count", 0)
            tasks_count = summary.get("tasks") if summary.get("tasks") is not None else 0
            images_count = summary.get("images", 0)
            sections_done = summary.get("sections_done", 0)
            
            progress_html = f"""
            <div style="
                background: rgba(255, 255, 255, 0.02);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 14px;
                padding: 1.5rem;
                margin-top: 1rem;
                font-family: 'Courier',monospace;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2), inset 0 0 15px rgba(255, 255, 255, 0.01);
                backdrop-filter: blur(20px);
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255, 255, 255, 0.06); padding-bottom: 0.6rem; margin-bottom: 1.2rem;">
                    <span style="color: #ffffff; font-weight: 600; font-size: 0.95rem; letter-spacing: 0.5px;">📡 AGENT CORE TELEMETRY</span>
                    <span style="color: #a5b4fc; font-size: 0.8rem; font-weight: 500;">STREAMING ACTIVE</span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem;">
                    <div style="background: rgba(0, 0, 0, 0.15); padding: 0.8rem; border-radius: 10px; border: 1px solid rgba(255,255,255,0.04); box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                        <span style="color: #94a3b8; font-size: 0.75rem; display: block; margin-bottom: 4px;">EXECUTION MODE</span>
                        <span style="color: #e2e8f0; font-size: 1.05rem; font-weight: 600;">{mode_val}</span>
                    </div>
                    <div style="background: rgba(0, 0, 0, 0.15); padding: 0.8rem; border-radius: 10px; border: 1px solid rgba(255,255,255,0.04); box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                        <span style="color: #94a3b8; font-size: 0.75rem; display: block; margin-bottom: 4px;">RESEARCH ACTIVE</span>
                        <span style="color: #e2e8f0; font-size: 1.05rem; font-weight: 600;">{needs_research_val}</span>
                    </div>
                    <div style="background: rgba(0, 0, 0, 0.15); padding: 0.8rem; border-radius: 10px; border: 1px solid rgba(255,255,255,0.04); box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                        <span style="color: #94a3b8; font-size: 0.75rem; display: block; margin-bottom: 4px;">SEARCH QUERIES</span>
                        <span style="color: #e2e8f0; font-size: 1.05rem; font-weight: 600;">{queries_count}</span>
                    </div>
                    <div style="background: rgba(0, 0, 0, 0.15); padding: 0.8rem; border-radius: 10px; border: 1px solid rgba(255,255,255,0.04); box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                        <span style="color: #94a3b8; font-size: 0.75rem; display: block; margin-bottom: 4px;">EVIDENCE COUNT</span>
                        <span style="color: #e2e8f0; font-size: 1.05rem; font-weight: 600;">{evidence_count} items</span>
                    </div>
                    <div style="background: rgba(0, 0, 0, 0.15); padding: 0.8rem; border-radius: 10px; border: 1px solid rgba(255,255,255,0.04); box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                        <span style="color: #94a3b8; font-size: 0.75rem; display: block; margin-bottom: 4px;">PLAN TASKS</span>
                        <span style="color: #e2e8f0; font-size: 1.05rem; font-weight: 600;">{tasks_count} tasks</span>
                    </div>
                    <div style="background: rgba(0, 0, 0, 0.15); padding: 0.8rem; border-radius: 10px; border: 1px solid rgba(255,255,255,0.04); box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                        <span style="color: #94a3b8; font-size: 0.75rem; display: block; margin-bottom: 4px;">SECTIONS WRITTEN</span>
                        <span style="color: #e2e8f0; font-size: 1.05rem; font-weight: 600;">{sections_done} sections</span>
                    </div>
                </div>
            </div>
            """
            progress_area.markdown(progress_html, unsafe_allow_html=True)

            log(f"[{kind}] {json.dumps(payload, default=str)[:1200]}")

        elif kind == "final":
            out = payload
            st.session_state["last_out"] = out
            # Clear cached profile data so they fetch the newly created blog
            st.session_state.pop("profile_stats", None)
            st.session_state.pop("profile_blogs", None)
            status.update(label="✅ Done", state="complete", expanded=False)
            log("[final] received final state")

# Render last result (if any)
out = st.session_state.get("last_out")
if out:
    # --- Plan tab ---
    with tab_plan:
        st.markdown("<h3 style='color:#a5b4fc;'>🧩 Blog Plan</h3>", unsafe_allow_html=True)
        plan_obj = out.get("plan")
        if not plan_obj:
            st.info("No plan found in output.")
        else:
            if hasattr(plan_obj, "model_dump"):
                plan_dict = plan_obj.model_dump()
            elif isinstance(plan_obj, dict):
                plan_dict = plan_obj
            else:
                plan_dict = json.loads(json.dumps(plan_obj, default=str))

            st.write("**Title:**", plan_dict.get("blog_title"))
            cols = st.columns(3)
            cols[0].write("**Audience:** " + str(plan_dict.get("audience")))
            cols[1].write("**Tone:** " + str(plan_dict.get("tone")))
            cols[2].write("**Blog kind:** " + str(plan_dict.get("blog_kind", "")))

            tasks = plan_dict.get("tasks", [])
            if tasks:
                df = pd.DataFrame(
                    [
                        {
                            "id": t.get("id"),
                            "title": t.get("title"),
                            "target_words": t.get("target_words"),
                            "requires_research": t.get("requires_research"),
                            "requires_citations": t.get("requires_citations"),
                            "requires_code": t.get("requires_code"),
                            "tags": ", ".join(t.get("tags") or []),
                        }
                        for t in tasks
                    ]
                ).sort_values("id")
                st.dataframe(df, use_container_width=True, hide_index=True)

                with st.expander("Task details"):
                    st.json(tasks)

    # --- Evidence tab ---
    with tab_evidence:
        st.markdown("<h3 style='color:#a5b4fc;'>🔎 Research Evidence</h3>", unsafe_allow_html=True)
        evidence = out.get("evidence") or []
        if not evidence:
            st.info("No evidence returned (maybe closed_book mode or no Tavily key/results).")
        else:
            rows = []
            for e in evidence:
                if hasattr(e, "model_dump"):
                    e = e.model_dump()
                rows.append(
                    {
                        "title": e.get("title"),
                        "published_at": e.get("published_at"),
                        "source": e.get("source"),
                        "url": e.get("url"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Preview tab ---
    with tab_preview:
        st.markdown("<h3 style='color:#a5b4fc;'>📝 Article Preview</h3>", unsafe_allow_html=True)
        final_md = out.get("final") or ""
        if not final_md:
            st.warning("No final markdown found.")
        else:
            render_markdown_with_local_images(final_md)

            plan_obj = out.get("plan")
            if hasattr(plan_obj, "blog_title"):
                blog_title = plan_obj.blog_title
            elif isinstance(plan_obj, dict):
                blog_title = plan_obj.get("blog_title", "blog")
            else:
                # fallback: parse from markdown title
                blog_title = extract_title_from_md(final_md, "blog")

            md_filename = f"{safe_slug(blog_title)}.md"
            st.download_button(
                "⬇️ Download Markdown",
                data=final_md.encode("utf-8"),
                file_name=md_filename,
                mime="text/markdown",
            )

            bundle = bundle_zip(final_md, md_filename, Path("images"))
            st.download_button(
                "📦 Download Bundle (MD + images)",
                data=bundle,
                file_name=f"{safe_slug(blog_title)}_bundle.zip",
                mime="application/zip",
            )

    # --- Images tab ---
    with tab_images:
        st.markdown("<h3 style='color:#a5b4fc;'>🖼️ Generated Images</h3>", unsafe_allow_html=True)
        specs = out.get("image_specs") or []
        images_dir = Path("images")

        if not specs and not images_dir.exists():
            st.info("No images generated for this blog.")
        else:
            if specs:
                st.write("**Image plan:**")
                st.json(specs)

            if images_dir.exists():
                files = [p for p in images_dir.iterdir() if p.is_file()]
                if not files:
                    st.warning("images/ exists but is empty.")
                else:
                    for p in sorted(files):
                        st.image(str(p), caption=p.name, use_container_width=True)

                z = images_zip(images_dir)
                if z:
                    st.download_button(
                        "⬇️ Download Images (zip)",
                        data=z,
                        file_name="images.zip",
                        mime="application/zip",
                    )

    # --- Logs tab ---
    with tab_logs:
        st.markdown("<h3 style='color:#a5b4fc;'>🧾 Event Logs</h3>", unsafe_allow_html=True)
        if "logs" not in st.session_state:
            st.session_state["logs"] = []
        if logs:
            st.session_state["logs"].extend(logs)

        st.text_area("Event log", value="\n\n".join(st.session_state["logs"][-80:]), height=520)
else:
    st.markdown("""
    <div class="cyber-ready-card">
      <div class="cyber-ready-icon">✍️</div>
      <h3 class="cyber-ready-title">Awaiting Topic Input</h3>
      <p style="font-family: 'Courier',monospace; color: #94a3b8; font-size: 0.95rem; margin-bottom: 1.5rem;">
        Please specify a topic in the side console to initialize the neural compiler.
      </p>
      <div style="
          display: inline-block;
          padding: 0.5rem 1.2rem;
          background: rgba(255, 255, 255, 0.02);
          border: 1px dashed rgba(255, 255, 255, 0.12);
          border-radius: 8px;
          font-size: 0.85rem;
          color: #94a3b8;
          font-family: 'Courier',monospace;
      ">
        STANDBY MODE
      </div>
    </div>
    """, unsafe_allow_html=True)