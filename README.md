# DeepWrite ✍️
### The Agentic Planning & Research Engine

> A multi-agent AI pipeline that researches, plans, writes, fact-checks, and SEO-audits a full technical blog post — end to end — in under 90 seconds.

Built for **AMS 691: Foundations and Frontiers of Large Language Models** at Stony Brook University.

---

## What makes this different

Every existing AI writing tool is a single LLM call dressed up with a nice UI. DeepWrite is a **9-node orchestrated pipeline** where each agent has one job:

```
Topic → Router → Memory → (Research?) → Orchestrator
      → Workers + Critic loop (parallel)
      → Reducer → Fact-Checker → SEO Audit
```

| Node | Role | What it does |
|---|---|---|
| 🔀 **Router** | Classifier | Decides if live web research is needed |
| 🧠 **Memory** | Context retriever | Fetches your past articles to match writing style |
| 🔍 **Research** | Web agent | Queries Tavily for real sources |
| 📋 **Orchestrator** | Editor-in-Chief | Builds structured plan: sections, goals, word targets |
| ✍️ **Workers** | Parallel writers | Multiple agents write sections simultaneously |
| 🎯 **Critic** | Quality gate | Scores each section 0–10; sends weak ones back for revision |
| 🔗 **Reducer** | Merger | Combines sections into one coherent article |
| ✅ **Fact-Checker** | Verifier | Cross-references every claim against evidence |
| 📈 **SEO Audit** | Optimiser | Scores 0–100, suggests keywords and meta description |

---

## Features

- **Live pipeline tracker** — watch every node complete in real time with timing
- **Critic–revision loop** — sections scoring below 6.5/10 are automatically revised
- **Writer memory** — ChromaDB stores past articles; your style improves over time
- **AI Editor** — chat with your article: edit, ask questions, use quick actions
- **Undo stack** — roll back any AI edit instantly
- **Draft history** — every run auto-saved to SQLite; reload any past article
- **HTML export** — styled, self-contained single-file with syntax highlighting
- **SEO audit panel** — score, keywords, meta description, issue list by severity
- **Fact-check panel** — per-claim verdicts with confidence bars and source links

---

## Tech stack

| Layer | Tech |
|---|---|
| Pipeline | LangGraph |
| LLM | Llama 3.3 70B via Groq |
| Web search | Tavily API |
| Vector memory | ChromaDB + sentence-transformers |
| Frontend | Streamlit |
| Draft storage | SQLite |

---

## Quick start

### 1. Clone
```bash
git clone https://github.com/iMeet07/deepwrite.git
cd deepwrite
```

### 2. Install
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set up API keys
Create a `.env` file in the project root:
```env
XAI_API_KEY=gsk_...          # Groq API key — free at console.groq.com
TAVILY_API_KEY=tvly-...      # Web search — free tier at tavily.com
```

### 4. Run
```bash
streamlit run bwa_frontend.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## How the critic loop works

```
Write section
     ↓
Critic scores: accuracy / depth / clarity / grounding (0–10 each)
     ↓
overall ≥ 6.5?  →  YES  →  Accept
     ↓ NO
Specific revision feedback generated
     ↓
Worker rewrites addressing every point
     ↓
Score again  →  Accept (max 2 attempts)
```

---

## How writer memory works

```
Article finished → Embedded → Stored in ChromaDB

Next run:
Topic → Retrieve top-3 similar past articles
      → Style context injected into every worker prompt
      → Output matches your established voice
```

---

## Project structure

```
deepwrite/
├── bwa_backend.py      # Full pipeline — all nodes + memory + critic
├── bwa_frontend.py     # Streamlit UI — tabs, AI editor, draft history
├── requirements.txt
├── .env                # API keys (not committed)
└── output/
    ├── *.md            # Generated articles
    ├── history.db      # SQLite draft history
    └── memory/         # ChromaDB writer memory
```

---

## Configuration

```python
class Settings:
    GROQ_MODEL             = "llama-3.3-70b-versatile"
    CRITIC_THRESHOLD       = 6.5   # sections below this get revised
    CRITIC_MAX_REVISIONS   = 2     # max revision attempts per section
    MEMORY_TOP_K           = 3     # past articles retrieved for style
    MAX_RESEARCH_QUERIES   = 2
    MAX_EVIDENCE_ITEMS     = 5
```

---

*117291342 · Brahmbhatt Meet Naresh · Stony Brook University · AMS 691 Spring 2026*
