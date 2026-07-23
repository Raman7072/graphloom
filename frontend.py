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
def load_css(css_file: str = "styles/style.css"):
    """Load external CSS file and inject into Streamlit DOM."""
    css_path = Path(css_file)
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def show_loading_screen(message: str, subtitle: str):
    """Render a premium liquid glass loading/teleporting overlay."""
    st.markdown(f"""
    <div style="display:flex; justify-content:center; align-items:center; height:80vh; font-family:'Courier',monospace; color:#cbd5e1;">
        <div style="text-align:center; background:rgba(255,255,255,0.02); padding:2.5rem; border:1px solid rgba(255,255,255,0.08); border-radius:20px; backdrop-filter:blur(32px); box-shadow:0 8px 32px rgba(0,0,0,0.37);">
            <div style="font-size:1.8rem; font-weight:700; margin-bottom:0.8rem; letter-spacing:3px; background:linear-gradient(135deg,#a5b4fc 0%,#2dd4bf 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; animation:pulse 1.2s infinite;">{message}</div>
            <div style="font-size:0.8rem; color:#64748b; text-transform:uppercase; letter-spacing:2px;">{subtitle}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path, allowed_filenames: Optional[set] = None) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))

        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    if allowed_filenames is not None and p.name not in allowed_filenames:
                        continue
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def images_zip(images_dir: Path, allowed_filenames: Optional[set] = None) -> Optional[bytes]:
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                if allowed_filenames is not None and p.name not in allowed_filenames:
                    continue
                z.write(p, arcname=str(p))
    return buf.getvalue()


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Stream graph progress if available; else invoke.
    Yields ("updates"/"values"/"final", payload).

    IMPORTANT: Never call .invoke() after .stream() — stream() already runs
    the full graph. Calling invoke() again would generate a second blog.
    Instead, capture the final state from the last streamed step.
    """
    # ── mode 1: stream updates ────────────────────────────────────────────────
    try:
        last_state: Dict[str, Any] = {}
        for step in graph_app.stream(inputs, stream_mode="values"):
            yield ("updates", step)
            last_state = step  # each "values" step IS the full state at that point
        if last_state:
            yield ("final", last_state)
            return
    except Exception:
        pass

    # ── fallback: plain invoke ────────────────────────────────────────────────
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
            st.image(src, caption=caption or (alt or None), width="stretch")
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists():
                st.image(str(img_path), caption=caption or (alt or None), width="stretch")
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
    page_title="GraphLoom",
    page_icon="styles/images/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load external CSS styles
load_css("styles/style.css")

# Initialise DB tables (no-op if DB not configured)
if _AUTH_AVAILABLE:
    try:
        init_db()
    except Exception:
        pass

# ── Cookie manager (must be initialised before any st.stop()) ────
_cookie_manager = stx.CookieManager(key="graphloom_cm") if _COOKIES_AVAILABLE else None

# Handle pending cookie deletion on logout
if _COOKIES_AVAILABLE and _cookie_manager and st.session_state.get("logout_pending"):
    st.session_state.pop("logout_pending", None)
    try:
        _cookie_manager.delete("graphloom_session", key="logout_delete_cookie")
        _cookie_manager.delete("graphloom_page", key="logout_delete_page_cookie")
    except Exception:
        pass

# Auto-restore session from cookie on page load / refresh
if _AUTH_AVAILABLE and _COOKIES_AVAILABLE and "user" not in st.session_state:
    if not st.session_state.get("logged_out", False):
        cookies = _cookie_manager.get_all()
        _token = cookies.get("graphloom_session") if cookies else None

        if _token:
            _user_from_cookie = verify_session_token(str(_token))
            if _user_from_cookie:
                st.session_state["user"] = _user_from_cookie
                st.session_state["page"] = cookies.get("graphloom_page", "home")
                st.rerun()

        # If cookies haven't been checked for this fresh browser session yet,
        # render the loading screen and stop so the CookieManager iframe HTML/JS
        # reaches the browser to send document.cookie back to Streamlit.
        if not st.session_state.get("_cookie_checked", False):
            st.session_state["_cookie_checked"] = True
            show_loading_screen("RESTORING SESSION...", "Verifying authentication...")
            st.stop()


# ── Auth helpers ─────────────────────────────────────────────
def _render_auth_page():
    """Full-page login / register UI shown when user is not logged in."""
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center; padding:2.5rem 0 1.5rem 0;">
            <h1 style="font-family:'Courier',monospace; font-size:2.4rem; font-weight:800;
                background:linear-gradient(135deg,#a5b4fc 0%,#818cf8 50%,#2dd4bf 100%);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                background-clip:text; margin:0 0 0.3rem 0;">GraphLoom</h1>
            <p style="color:#94a3b8; font-family:'Courier',monospace; font-size:0.9rem;
                letter-spacing:1px; margin:0;">GRAPH-POWERED BLOG GENERATION ENGINE</p>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_reg = st.tabs(["LOGIN", "REGISTER"])

        with tab_login:
            if "login_error" in st.session_state:
                st.error(st.session_state.pop("login_error"))
            with st.form("login_form", clear_on_submit=False):
                email_in = st.text_input("Email", placeholder="you@example.com")
                pass_in  = st.text_input("Password", type="password", placeholder="********")
                if st.form_submit_button("Login", width="stretch", type="primary"):
                    if not email_in or not pass_in:
                        st.error("Please fill in all fields.")
                    else:
                        st.session_state["login_pending"] = {"email": email_in, "password": pass_in}
                        st.rerun()

        with tab_reg:
            if "register_error" in st.session_state:
                st.error(st.session_state.pop("register_error"))
            with st.form("register_form", clear_on_submit=False):
                name_in    = st.text_input("Full Name", placeholder="Steve Rogers")
                email_in2  = st.text_input("Email", placeholder="you@example.com", key="reg_email")
                pass_in2   = st.text_input("Password", type="password",
                                           placeholder="Min 6 characters", key="reg_pass")
                confirm_in = st.text_input("Confirm Password", type="password",
                                           placeholder="Repeat password", key="reg_confirm")
                if st.form_submit_button("Create Account", width="stretch", type="primary"):
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

    # ── Back navigation ──────────────────────────────────────
    if st.button("BACK TO GRAPHLOOM", key="back_from_profile", width="stretch"):
        st.session_state["back_to_home_pending"] = True
        if _COOKIES_AVAILABLE and _cookie_manager:
            _cookie_manager.set(
                "graphloom_page",
                "home",
                max_age=30 * 24 * 3600,
                key="back_home_page_cookie_set"
            )
        st.rerun()

    # ── Hero profile card ─────────────────────────────────────
    st.markdown(f"""
    <div class="pg-hero">
        <div class="pg-avatar">{initials}</div>
        <div style="position:relative;z-index:1;">
            <div class="pg-name">{stats['name']}</div>
            <div class="pg-email">{stats['email']}</div>
            <span class="pg-badge">MEMBER SINCE {ms_str.upper()}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Stat cards ────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "DOCS", str(stats["blog_count"]), "Blogs Generated"),
        (c2, "IMGS", str(stats["image_count"]), "Images Created"),
        (c3, "DAYS", str(days_active), "Days Active"),
        (c4, "SECT", str(stats["blog_count"] * 6), "Sections Written"),
    ]
    for col, tag, num, label in cards:
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <span class="stat-tag">{tag}</span>
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

            with st.expander("Change Display Name", expanded=False):
                with st.form("change_name_form"):
                    new_name = st.text_input("New Name", value=stats["name"])
                    if st.form_submit_button("Update Name", type="primary", width="stretch"):
                        if update_user_name(uid, new_name):
                            st.session_state["user"]["name"] = new_name.strip()
                            st.session_state.pop("profile_stats", None)
                            st.success("Name updated!")
                            st.rerun()
                        else:
                            st.error("Name cannot be empty.")

            with st.expander("Change Password", expanded=False):
                with st.form("change_pass_form"):
                    old_p = st.text_input("Current Password", type="password")
                    new_p = st.text_input("New Password", type="password", placeholder="Min 6 characters")
                    cnf_p = st.text_input("Confirm New Password", type="password")
                    if st.form_submit_button("Update Password", type="primary", width="stretch"):
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
            <div class="danger-header pg-section-title" style="color:#f87171; margin-bottom:0.3rem;">Danger Zone</div>
            <div style="font-family:'Courier Prime', 'Courier', monospace; font-size:0.8rem; color:#64748b; margin-bottom:0.8rem;">
                Permanently deletes your account and all generated blogs.
            </div>
            """, unsafe_allow_html=True)
            with st.expander("Delete My Account", expanded=False):
                with st.form("delete_acc_form"):
                    confirm_pass = st.text_input("Enter your password to confirm", type="password")
                    if st.form_submit_button("Delete Account Forever", width="stretch"):
                        res = delete_user_account(uid, confirm_pass)
                        if res == "ok":
                            st.session_state.clear()
                            st.success("Account deleted.")
                            st.rerun()
                        else:
                            st.error(res)




# Intercept pending auth database load actions (shows loading screen while querying)
if "login_pending" in st.session_state:
    show_loading_screen("Check Post!!!", "Verifying Credentials")
    pending = st.session_state.pop("login_pending")
    try:
        user = login_user(pending["email"], pending["password"])
        if user:
            st.session_state["user"] = user
            st.session_state["logged_out"] = False
            if _COOKIES_AVAILABLE and _cookie_manager:
                _cookie_manager.set(
                    "graphloom_session",
                    create_session_token(user),
                    max_age=30 * 24 * 3600,
                    key="login_cookie_set"
                )
                _cookie_manager.set(
                    "graphloom_page",
                    "home",
                    max_age=30 * 24 * 3600,
                    key="login_page_cookie_set"
                )
            st.session_state["page"] = "home"
        else:
            st.session_state["login_error"] = "Invalid email or password."
    except Exception as e:
        st.session_state["login_error"] = f"Login failed: {e}"
    st.rerun()

if "register_pending" in st.session_state:
    show_loading_screen("H! HUMAN...", "Creating Secure Account")
    pending = st.session_state.pop("register_pending")
    try:
        result = register_user(pending["name"], pending["email"], pending["password"])
        if isinstance(result, dict):
            st.session_state["user"] = result
            st.session_state["logged_out"] = False
            if _COOKIES_AVAILABLE and _cookie_manager:
                _cookie_manager.set(
                    "graphloom_session",
                    create_session_token(result),
                    max_age=30 * 24 * 3600,
                    key="register_cookie_set"
                )
                _cookie_manager.set(
                    "graphloom_page",
                    "home",
                    max_age=30 * 24 * 3600,
                    key="register_page_cookie_set"
                )
            st.session_state["page"] = "home"
        else:
            st.session_state["register_error"] = str(result)
    except Exception as e:
        st.session_state["register_error"] = f"Registration failed: {e}"
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
    show_loading_screen("RETURNING...", "Loading Dashboard Matrix")
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
# Embed logo as base64 background inside the header card
import base64 as _b64

@st.cache_data
def _load_logo_b64() -> str:
    _logo_path = Path("styles/images/graphloom_logo.png")
    if _logo_path.exists():
        with open(_logo_path, "rb") as _f:
            return _b64.b64encode(_f.read()).decode()
    return ""

_logo_b64 = _load_logo_b64()

_logo_bg_css = ""
if _logo_b64:
    _logo_bg_css = f"""
<style>
.cyber-header {{
    background-image: url("data:image/png;base64,{_logo_b64}") !important;
    background-size: 100% auto !important;
    background-repeat: no-repeat !important;
    background-position: center center !important;
}}
</style>
"""

st.markdown(_logo_bg_css + """
<div class="cyber-header"></div>
""", unsafe_allow_html=True)


with st.sidebar:
    # ── User info + logout ────────────────────────────────
    if _current_user:
        if st.button("MY PROFILE", width="stretch"):
            st.session_state["profile_pending"] = True
            if _COOKIES_AVAILABLE and _cookie_manager:
                _cookie_manager.set(
                    "graphloom_page",
                    "profile",
                    max_age=30 * 24 * 3600,
                    key="profile_page_cookie_set"
                )
            st.rerun()
        if st.button("LOGOUT", width="stretch"):
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
      GENERATE NEW BLOG
    </div>
    """, unsafe_allow_html=True)

    topic = st.text_area(
        "TOPIC",
        height=120,
        placeholder="e.g. How Transformer attention works…",
    )
    as_of = st.date_input("AS OF DATE", value=date.today())
    run_btn = st.button("GENERATE BLOG", type="primary", width="stretch")

    st.divider()
    st.markdown("""
    <div style="font-family:'Courier',monospace; font-size:0.9rem; font-weight:600;
        color:#e2e8f0; margin-bottom:0.5rem; letter-spacing:0.5px; text-transform:uppercase;">
        MY BLOGS
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

        if st.button("LOAD SELECTED BLOG", width="stretch"):
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
                        "image_specs": [{"filename": _img["filename"], "alt": "Loaded Image", "caption": _img["filename"]} for _img in _blog.get("images", [])],
                        "final": _blog["content"],
                    }
                    st.success("Blog loaded.")
                    st.rerun()

    st.divider()
    st.markdown("© 2026 GraphLoom — All rights reserved.")
    



# Storage for latest run
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

# Layout
tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs(
    ["PLAN", "EVIDENCE", "MARKDOWN PREVIEW", "IMAGES", "LOGS"]
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

            _plan_val = current_state.get("plan")
            _tasks_cnt = None
            if _plan_val:
                if hasattr(_plan_val, "tasks"):
                    _tasks_cnt = len(_plan_val.tasks)
                elif isinstance(_plan_val, dict):
                    _tasks_cnt = len(_plan_val.get("tasks", []))

            summary = {
                "mode": current_state.get("mode"),
                "needs_research": current_state.get("needs_research"),
                "queries": current_state.get("queries", [])[:5] if isinstance(current_state.get("queries"), list) else [],
                "evidence_count": len(current_state.get("evidence", []) or []),
                "tasks": _tasks_cnt,
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
                    <span style="color: #f5f5f5; font-weight: 600; font-size: 0.95rem; letter-spacing: 0.5px;">📡 AGENT CORE TELEMETRY</span>
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
            st.session_state["logs"] = st.session_state.get("logs", []) + logs
            status.update(label="✅ Done", state="complete", expanded=False)
            log("[final] received final state")
            # Force a clean rerun so all tabs render properly from session state
            st.rerun()

# Render last result (if any)
out = st.session_state.get("last_out")
if out:
    specs = out.get("image_specs") or []
    allowed_filenames = {s["filename"] for s in specs if isinstance(s, dict) and "filename" in s}
    # --- Plan tab ---
    with tab_plan:
        st.markdown("<div class='cyber-tab-header'>BLOG PLAN</div>", unsafe_allow_html=True)
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
                st.dataframe(df, width="stretch", hide_index=True)

                with st.expander("Task details"):
                    st.json(tasks)

    # --- Evidence tab ---
    with tab_evidence:
        st.markdown("<div class='cyber-tab-header'>RESEARCH EVIDENCE</div>", unsafe_allow_html=True)
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
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # --- Preview tab ---
    with tab_preview:
        st.markdown("<div class='cyber-tab-header'>ARTICLE PREVIEW</div>", unsafe_allow_html=True)
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
                "DOWNLOAD MARKDOWN",
                data=final_md.encode("utf-8"),
                file_name=md_filename,
                mime="text/markdown",
                width="stretch",
            )

            bundle = bundle_zip(final_md, md_filename, Path("images"), allowed_filenames)
            st.download_button(
                "DOWNLOAD BUNDLE (MD + IMAGES)",
                data=bundle,
                file_name=f"{safe_slug(blog_title)}_bundle.zip",
                mime="application/zip",
                width="stretch",
            )

    # --- Images tab ---
    with tab_images:
        st.markdown("<div class='cyber-tab-header'>GENERATED IMAGES</div>", unsafe_allow_html=True)
        images_dir = Path("images")

        if not allowed_filenames:
            st.info("No images generated for this blog.")
        else:
            detailed_specs = [s for s in specs if isinstance(s, dict) and "prompt" in s]
            if detailed_specs:
                st.write("**Image plan:**")
                st.json(detailed_specs)

            if images_dir.exists():
                files = [p for p in images_dir.iterdir() if p.is_file() and p.name in allowed_filenames]
                if not files:
                    st.warning("No image files found on disk for this blog.")
                else:
                    for p in sorted(files):
                        st.image(str(p), caption=p.name, width="stretch")

                z = images_zip(images_dir, allowed_filenames)
                if z:
                    st.download_button(
                        "DOWNLOAD IMAGES (ZIP)",
                        data=z,
                        file_name="images.zip",
                        mime="application/zip",
                        width="stretch",
                    )

    # --- Logs tab ---
    with tab_logs:
        st.markdown("<div class='cyber-tab-header'>EVENT LOGS</div>", unsafe_allow_html=True)
        if "logs" not in st.session_state:
            st.session_state["logs"] = []
        if logs:
            st.session_state["logs"].extend(logs)

        st.text_area("Event log", value="\n\n".join(st.session_state["logs"][-80:]), height=520)
else:
    st.markdown("""
    <div class="cyber-ready-card">
      <div class="cyber-ready-icon" style="color: #6366f1; font-family: 'Courier Prime', 'Courier', monospace; font-size: 3rem; letter-spacing: -2px; margin-bottom: 1.5rem;">[&gt;_]</div>
      <h3 class="cyber-ready-title">AWAITING TOPIC INPUT</h3>
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