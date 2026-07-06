# ✍️ Blog Writing Agent

A multi-agent AI pipeline that researches, plans, writes, and illustrates technical blog posts — fully automated using **LangGraph**, **Groq**, **Tavily** and **HuggingFace**.

---

## 🧠 How It Works

The agent runs as a **LangGraph state machine** with the following pipeline:

```
Topic Input
    │
    ▼
┌─────────┐     closed_book     ┌──────────────┐
│ Router  │ ──────────────────► │ Orchestrator │
│  Node   │                     │  (Planner)   │
└─────────┘                     └──────┬───────┘
    │ needs_research = true             │
    ▼                                   │ Fan-out
┌──────────┐                    ┌───────▼──────┐
│ Research │ ──────────────────►│   Workers    │ (parallel section writers)
│  (Tavily)│                    └───────┬──────┘
└──────────┘                            │
                                        ▼
                               ┌─────────────────┐
                               │ Reducer Subgraph │
                               │  merge_content   │
                               │  decide_images   │
                               │  generate_images │
                               └────────┬─────────┘
                                        │
                                        ▼
                                  Final Blog (MD)
```

### Agent Nodes

| Node | Role |
|---|---|
| **Router** | Decides if web research is needed (`closed_book` / `hybrid` / `open_book`) |
| **Research** | Runs Tavily searches, LLM-filters results into structured `EvidenceItem` objects |
| **Orchestrator** | Produces a structured `Plan` — title, audience, tone, 5–9 section tasks |
| **Workers** | Each task runs in parallel; writes one markdown section with optional code + citations |
| **merge_content** | Sorts and joins all worker sections into a single markdown document |
| **decide_images** | LLM decides where images help; inserts `[[IMAGE_N]]` placeholders |
| **generate_images** | Calls HuggingFace (fal-ai) to generate images and replaces placeholders |

---

## 🚀 Getting Started

### 1. Clone & set up environment

```bash
git clone [https://github.com/Raman7072/Blog_writing_agent.git](https://github.com/Raman7072/Blog_writing_agent.git)
cd "blog writing agent"

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...          # Optional — enables web research
HUGGINGFACEHUB_API_TOKEN=hf_... # Optional — enables image generation
```

> ⚠️ Never commit your `.env` file. It is already listed in `.gitignore`.

### 3. Run the app

```bash
streamlit run frontend.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🖥️ UI Overview

The Streamlit frontend provides:

- **Sidebar** — Enter a topic, set an as-of date, and click 🚀 Generate Blog
- **Past Blogs** — Load any previously generated `.md` file from disk
- **🧩 Plan tab** — View the structured blog plan (title, audience, tone, tasks table)
- **🔎 Evidence tab** — Browse research sources gathered by Tavily
- **📝 Preview tab** — Rendered markdown with images + download buttons
- **🖼️ Images tab** — View all generated images; download as zip
- **🧾 Logs tab** — Full event log for debugging each graph step

---

## 📁 Project Structure

```
blog writing agent/
├── backend.py          # LangGraph agent — all nodes, schemas, graph definition
├── frontend.py         # Streamlit UI
├── requirements.txt    # Python dependencies
├── .env                # API keys (gitignored)
├── .gitignore
├── images/             # Generated images saved here (gitignored)
└── *.md                # Generated blog posts saved here
```

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM | [Groq](https://groq.com) — `llama-3.3-70b-versatile` |
| Web Research | [Tavily Search](https://tavily.com) |
| Image Generation | [HuggingFace InferenceClient](https://huggingface.co/docs/huggingface_hub) via `fal-ai` — `krea/Krea-2-Turbo` |
| Frontend | [Streamlit](https://streamlit.io) |
| Data Validation | [Pydantic v2](https://docs.pydantic.dev) |

---

## 🔑 API Keys

| Key | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | LLM for all text generation |
| `TAVILY_API_KEY` | ⚡ Optional | Web research (hybrid / open_book mode) |
| `HUGGINGFACEHUB_API_TOKEN` | 🖼️ Optional | AI image generation |

> Without Tavily, the agent runs in `closed_book` mode (evergreen topics only).  
> Without HuggingFace, image placeholders are left as blockquote error notes in the markdown.

---

## 📤 Output

Each generated blog is saved as a `.md` file in the working directory (e.g. `intro_to_transformers.md`).  
Generated images are saved under `images/`.

You can also download directly from the **Preview tab**:
- `⬇️ Download Markdown` — the `.md` file
- `📦 Download Bundle` — markdown + images as a `.zip`

---

## 🌐 Deployment

### Streamlit Community Cloud (Recommended — Free)

1. Push to a public GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New App
3. Set `frontend.py` as the entry point
4. Add your API keys under **Settings → Secrets**:
   ```toml
   GROQ_API_KEY = "gsk_..."
   TAVILY_API_KEY = "tvly-..."
   HUGGINGFACEHUB_API_TOKEN = "hf_..."
   ```

### Railway / Render

Add a `Procfile`:
```
web: streamlit run frontend.py --server.port $PORT --server.address 0.0.0.0
```

---

## 📄 License

MIT
