# 🧶 GraphLoom

> **An Autonomous, Multi-Agent Technical Blog Generation Engine Powered by LangGraph, Groq and Streamlit.**

![GraphLoom Logo](images/logo.png)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-000000?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-f05138?style=for-the-badge)](https://groq.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)

---

## 📌 Overview

**GraphLoom** is an enterprise-grade, multi-agent AI system designed to automate deep research, structured outline planning, parallel section drafting, dynamic citation grounding, and technical diagram synthesis for publication-ready blog posts.

Unlike single-prompt LLM wrappers, GraphLoom models technical writing as a stateful, cyclical graph execution workflow using **LangGraph**. It routes content requirements dynamically, executes parallel web searches, fans out task workers across structured sub-sections, synthesizes AI visuals or frosted-glass diagrams, and persists all assets to PostgreSQL with a secure multi-user architecture.

---

## ✨ Key Features

- 🧠 **Adaptive Intelligence Routing**:
  - Dynamically classifies user topics into three operational modes:
    - **`closed_book`**: For evergreen concepts without external dependency.
    - **`hybrid`**: Merges internal knowledge with targeted, up-to-date web research.
    - **`open_book`**: Enforces strict temporal bounds (e.g. 7-day recency cutoff) for news & tech roundups.

- ⚡ **Parallel Web Research Engine**:
  - Leverages concurrent `Tavily` search workers (`ThreadPoolExecutor`) to gather authoritative evidence items.
  - Normalizes metadata, deduplicates canonical URLs, and computes published dates against recency thresholds.

- 🔀 **Parallel Map-Reduce Worker Fan-Out**:
  - **Orchestrator**: Deconstructs topics into 5–9 granular task modules with word-count targets, tags, and citation flags.
  - **Worker Nodes**: Uses LangGraph's dynamic `Send` API to parallelize section drafting, enforcing scope bounds and inline URL citations.
  - **Reducer Subgraph**: Merges draft sections, evaluates image needs, generates visual specs, and replaces placeholders with finalized images.

- 🎨 **Visual Asset Synthesis & Resilient Fallback Engine**:
  - Integrates Hugging Face (`FLUX.1-schnell`) for high-fidelity technical diagrams and AI artwork.
  - Includes a zero-downtime fallback engine using **Pillow** to render custom dark-mode glassmorphic architecture diagrams when API limits occur.

- 🔒 **Secure User Management & PostgreSQL Persistence**:
  - Built-in multi-user database storage with `bcrypt` password hashing and `ThreadedConnectionPool`.
  - Signed HMAC-SHA256 session cookies for seamless session persistence across refreshes.
  - User analytics: tracks document counts, image output, registration metrics, and interactive blog loading/deletion.

- 💎 **Cybernetic Glassmorphism Interface**:
  - Custom dark theme UI (`#0b0d19` palette) built with Streamlit and styled via modular CSS (`styles/style.css`).
  - Real-time agent telemetry stream dashboard providing visibility into node state transitions, search queries, active workers, and generated images.
  - One-click downloads for standalone Markdown files or complete ZIP archives (Markdown + image assets).

---

## 🏗️ System Architecture

```mermaid
graph TD
    %% Nodes
    User([User Prompt]) --> Router[📡 Router Node]
    
    Router -->|closed_book| Orch[📋 Orchestrator Node]
    Router -->|hybrid / open_book| Research[🔎 Research Node - Tavily API]
    Research --> Orch
    
    Orch -->|Dynamic Fan-Out via Send API| Workers[👷 Worker Nodes - Parallel Section Writers]
    
    subgraph Reducer Subgraph ["⚙️ Reducer Subgraph"]
        Merge[📝 Merge Content] --> DecideImg[🖼️ Decide Image Specs]
        DecideImg --> GenImg[⚡ Generate & Place Images]
    end
    
    Workers --> Merge
    GenImg --> DB[(🗄️ PostgreSQL Persistence)]
    GenImg --> UI[💻 Streamlit UI]
```

---

## 📂 Project Structure

```
GraphLoom/
├── backend.py            # LangGraph multi-agent backend state machine & node logic
├── frontend.py           # Streamlit UI, real-time telemetry dashboard & rendering
├── auth.py               # Database pool, bcrypt authentication, cookie tokens & profile stats
├── styles/
│   └── style.css         # Frosted glassmorphism dark aesthetic stylesheet
├── images/
│   ├── graphloom_logo.png   # Main application logo header
│   ├── graphloom_favicon.png# App favicon
│   ├── ss1.png              # UI telemetry interface screenshot
│   └── ss2.png              # Generated markdown & image preview screenshot
├── requirements.txt      # Project dependencies
├── .env                  # Environment key configuration
└── README.md             # Project documentation
```

---

## 🛠️ Tech Stack

- **Orchestration**: [LangGraph](https://langchain-ai.github.io/langgraph/), [LangChain Core](https://github.com/langchain-ai/langchain)
- **Language Models**: [Groq API](https://groq.com/) (`llama-3.3-70b-versatile`)
- **Research Tools**: [Tavily Search API](https://tavily.com/)
- **Image Generation**: [Hugging Face Inference API](https://huggingface.co/) (`black-forest-labs/FLUX.1-schnell`), [Pillow](https://python-pillow.org/)
- **Database & Auth**: [PostgreSQL](https://www.postgresql.org/), `psycopg2-binary`, `bcrypt`, `hmac`/`hashlib`
- **Frontend UI**: [Streamlit](https://streamlit.io/), `extra-streamlit-components`
- **Styling**: Vanilla CSS3 (Custom Glassmorphism Design System)

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python**: `3.10` or higher
- **PostgreSQL**: Local or hosted database instance (e.g., Supabase, Neon or local Postgres)

### 2. Installation

Clone the repository and enter the project directory:
```bash
git clone https://github.com/Raman7072/graphloom.git
cd graphloom
```

Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install required dependencies:
```bash
pip install -r requirements.txt
```

---

### 3. Environment Configuration

Create a `.env` file in the root directory:
```env
# LLM & Search APIs
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
HF_TOKEN=your_huggingface_token_here

# Database Configuration
DB_URL=postgresql://user:password@localhost:5432/graphloom_db

# Security & Session Secrets
SESSION_SECRET=your_random_hmac_secret_key_here
ENVIRONMENT=development

# LangSmith Tracking
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY=[LANGSMITH_API_KEY]
LANGSMITH_PROJECT='GraphLoom'
```

---

### 4. Database Setup

GraphLoom automatically initializes the required tables (`users`, `blogs`, `blog_images`) on first startup via `auth.init_db()`. Ensure your PostgreSQL database is reachable at the `DB_URL` provided in `.env`.

---

### 5. Running the Application

Launch the Streamlit web application:
```bash
streamlit run frontend.py
```

Open your browser at `http://localhost:8501` to access GraphLoom.

---

## 🖥️ UI Screenshots

| Real-Time Telemetry Stream | Generated Article & Diagrams |
| :---: | :---: |
| ![Telemetry Interface](images/ss1.png) | ![Markdown & Visual Preview](images/ss2.png) |

---

## 🔒 Security Best Practices

- Password hashes are stored using standard `bcrypt` key derivation with random salt.
- Session cookies use base64 HMAC-SHA256 signatures with timestamp expiration.
- Input fields sanitize database query parameters using `psycopg2` parameterized queries.
- Environment variables isolate private keys (`GROQ_API_KEY`, `TAVILY_API_KEY`, `HF_TOKEN`, `DB_URL`).

