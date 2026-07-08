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


# -----------------------------
# Helpers
# -----------------------------
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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Share+Tech+Mono&family=Inter:wght@300;400;500;600;700&display=swap');

/* Base Override */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #cbd5e1;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif;
    letter-spacing: 0.5px;
}

.stApp {
    background-color: #0b0d19 !important;
    background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.1) 0px, transparent 60%),
        radial-gradient(at 100% 0%, rgba(139, 92, 246, 0.1) 0px, transparent 60%),
        radial-gradient(at 50% 100%, rgba(45, 212, 191, 0.06) 0px, transparent 60%) !important;
    min-height: 100vh;
    position: relative;
    overflow-x: hidden;
}

/* Soothing floating color blobs */
.stApp::before {
    content: '';
    position: absolute;
    top: 5%;
    left: 8%;
    width: 450px;
    height: 450px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.14) 0%, transparent 70%);
    filter: blur(80px);
    pointer-events: none;
    z-index: 0;
    animation: blobFloat 22s infinite ease-in-out alternate;
}

.stApp::after {
    content: '';
    position: absolute;
    bottom: 10%;
    right: 8%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(45, 212, 191, 0.1) 0%, transparent 70%);
    filter: blur(80px);
    pointer-events: none;
    z-index: 0;
    animation: blobFloat 28s infinite ease-in-out alternate-reverse;
}

/* Glassmorphic Sidebar styling */
[data-testid="stSidebar"] {
    background: rgba(10, 15, 30, 0.55) !important;
    backdrop-filter: blur(25px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    box-shadow: 10px 0 30px rgba(0, 0, 0, 0.3) !important;
    z-index: 100;
}

/* Sidebar header */
.sidebar-header {
    font-family: 'Outfit', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: #f1f5f9;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 12px;
    margin-bottom: 22px;
    letter-spacing: 0.5px;
}

/* Sidebar Widget labels */
[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p,
label[data-testid="stWidgetLabel"] p {
    color: #cbd5e1 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px;
}

/* Text fields and Textareas - Frosted */
.stTextArea textarea, .stTextInput input, .stDateInput input {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.stTextArea textarea:focus, .stTextInput input:focus, .stDateInput input:focus {
    background: rgba(255, 255, 255, 0.04) !important;
    border-color: rgba(165, 180, 252, 0.4) !important; /* Soft Lavender */
    box-shadow: 0 0 15px rgba(165, 180, 252, 0.15) !important;
}

/* Primary generate button with soft gradient glow styling */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.75) 0%, rgba(45, 212, 191, 0.75) 100%) !important; /* Indigo to Teal */
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 12px !important;
    padding: 0.7rem 1.4rem !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.15) !important;
    cursor: pointer;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.25), 0 0 15px rgba(45, 212, 191, 0.15) !important;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.9) 0%, rgba(45, 212, 191, 0.9) 100%) !important;
    border-color: rgba(255, 255, 255, 0.25) !important;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"]:active {
    transform: translateY(0px) !important;
}

/* Secondary Button inside Sidebar (e.g. Load selected blog) */
[data-testid="stSidebar"] .stButton > button:not([kind="primary"]) {
    background: rgba(255, 255, 255, 0.03) !important;
    color: #cbd5e1 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer;
}

[data-testid="stSidebar"] .stButton > button:not([kind="primary"]):hover {
    background: rgba(255, 255, 255, 0.08) !important;
    color: #ffffff !important;
    border-color: rgba(255, 255, 255, 0.2) !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
}

/* Radio buttons list in sidebar */
[data-testid="stRadio"] label {
    font-family: 'Outfit', sans-serif !important;
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
    border-radius: 14px !important;
    padding: 6px !important;
    gap: 8px !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
    backdrop-filter: blur(10px) !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    padding: 0.6rem 1.3rem !important;
    transition: all 0.3s ease !important;
    border: 1px solid transparent !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #ffffff !important;
    background: rgba(255, 255, 255, 0.03) !important;
}

.stTabs [aria-selected="true"] {
    background: rgba(99, 102, 241, 0.12) !important;
    color: #ffffff !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.15) !important;
}

/* Frosted Glass Panels, Dataframes & Expanders */
.stDataFrame, [data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(20px) !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2) !important;
    transition: all 0.3s ease !important;
}

[data-testid="stExpander"]:hover {
    background: rgba(255, 255, 255, 0.03) !important;
    border-color: rgba(99, 102, 241, 0.2) !important;
    box-shadow: 0 8px 32px 0 rgba(99, 102, 241, 0.08) !important;
}

/* Status box override */
[data-testid="stStatus"] {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    color: #cbd5e1 !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15) !important;
}

/* Download buttons */
.stDownloadButton > button {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.3s ease !important;
}

.stDownloadButton > button:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    color: #ffffff !important;
    border-color: rgba(255, 255, 255, 0.2) !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15) !important;
}

/* Custom Alert styling */
.stAlert {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-left: 4px solid rgba(99, 102, 241, 0.8) !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.15) !important;
}

/* Text */
h1, h2, h3, h4 { color: #ffffff !important; }
p, li, label, span { color: #cbd5e1; }

/* Textarea specifically for event logs */
.stTextArea textarea {
    background: rgba(10, 15, 30, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #cbd5e1 !important;
    border-radius: 10px;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.82rem !important;
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
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 10px 40px rgba(0,0,0,0.2), inset 0 0 20px rgba(255,255,255,0.01);
    backdrop-filter: blur(25px);
    overflow: hidden;
    animation: slideUp 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}

.cyber-scanline {
    position: absolute;
    top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.03) 0%, transparent 60%);
    animation: rotateLiquid 20s linear infinite;
    pointer-events: none;
}

.cyber-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 3rem !important;
    font-weight: 800 !important;
    margin: 0 0 0.5rem 0 !important;
    background: linear-gradient(135deg, #a5b4fc 0%, #818cf8 50%, #2dd4bf 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    letter-spacing: 0.5px !important;
}

.cyber-subtitle {
    font-family: 'Outfit', sans-serif;
    font-weight: 400;
    font-size: 0.95rem;
    color: #94a3b8;
    margin-bottom: 1rem;
    letter-spacing: 1px;
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
    font-family: 'Outfit', sans-serif;
    font-weight: 500;
    font-size: 0.75rem;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: #cbd5e1;
    border-radius: 6px;
    letter-spacing: 0.5px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.cyber-ready-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 24px;
    padding: 3.5rem 2rem;
    text-align: center;
    max-width: 650px;
    margin: 3.5rem auto;
    box-shadow: 0 15px 45px rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(20px);
    animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}

.cyber-ready-icon {
    font-size: 4rem;
    margin-bottom: 1.5rem;
    filter: drop-shadow(0 4px 12px rgba(255, 255, 255, 0.15));
    animation: float 4s ease-in-out infinite;
}

.cyber-ready-title {
    font-family: 'Outfit', sans-serif;
    color: #ffffff;
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.75rem;
    letter-spacing: 0.5px;
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
    <div style="font-family:'Outfit', sans-serif; font-size:0.9rem; font-weight:600; color:#e2e8f0; margin-bottom:0.5rem; letter-spacing:0.5px; text-transform:uppercase;">
        📚 Past Databases
    </div>""", unsafe_allow_html=True)

    past_files = list_past_blogs()
    if not past_files:
        st.caption("No saved blogs found (*.md in current folder).")
        selected_md_file = None
    else:
        # Build labels from file name + (optional) parsed title
        options: List[str] = []
        file_by_label: Dict[str, Path] = {}
        for p in past_files[:50]:
            try:
                md_text = read_md_file(p)
                title = extract_title_from_md(md_text, p.stem)
            except Exception:
                title = p.stem
            label = f"{title}  ·  {p.name}"
            options.append(label)
            file_by_label[label] = p

        selected_label = st.radio(
            "Select a blog to load",
            options=options,
            index=0,
            label_visibility="collapsed",
        )
        selected_md_file = file_by_label.get(selected_label)

        if st.button("Load selected blog"):
            if selected_md_file:
                md_text = read_md_file(selected_md_file)
                # Load into session_state as if it were a run output
                st.session_state["last_out"] = {
                    "plan": None,          # old files don't include plan
                    "evidence": [],        # old files don't include evidence
                    "image_specs": [],     # optional (not persisted)
                    "final": md_text,      # markdown body
                }
                # also update the topic input to the title (best-effort) without changing UI
                st.session_state["topic_prefill"] = extract_title_from_md(md_text, selected_md_file.stem)

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
                font-family: 'Outfit', sans-serif;
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
      <p style="font-family: 'Outfit', sans-serif; color: #94a3b8; font-size: 0.95rem; margin-bottom: 1.5rem;">
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
          font-family: 'Outfit', sans-serif;
      ">
        STANDBY MODE
      </div>
    </div>
    """, unsafe_allow_html=True)