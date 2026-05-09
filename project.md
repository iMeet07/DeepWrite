---
title: "DeepWrite: The Agentic Planning & Research Engine"
author: "Brahmbhatt Meet Naresh"
sid: "117291342"
semester: "s26"
category: "developer-tools"
tags: ["multi-agent", "langgraph", "content-generation", "nlp", "rag"]
thumbnail: "https://drive.google.com/file/d/1gcT9Vu42t78BLpzFexai7Nr60_hiaue7/view?usp=sharing"
video: "https://drive.google.com/file/d/1UsR38H6tjVJFMIPJ9dHWXZYOjvL6B5pF/view?usp=sharing"
github: "https://github.com/iMeet07/deepwrite"
---

# DeepWrite: The Agentic Planning & Research Engine

A multi-agent AI pipeline that researches, plans, writes, fact-checks, and SEO-audits a full technical blog post — end to end — in under 90 seconds.

## Problem

Standard AI writing tools suffer from three fundamental limitations:

**The Hallucination Gap** — LLMs write in one sequential pass with no verification step, producing confident-sounding content that is factually wrong. There is no mechanism to cross-reference claims against real sources.

**The Research Wall** — Human writers spend 80% of their time on research and only 20% actually writing. Existing tools ignore research entirely, generating generic content that lacks grounding in real-world data.

**Linear Generation** — Every tool from ChatGPT to Jasper generates content in a single LLM call. This produces shallow, unfocused output because the model must simultaneously manage structure, research grounding, tone, and word count in one shot.

## Solution

DeepWrite is a **7-node LangGraph pipeline** that separates the writing process into specialist agents — the same way a professional newsroom works. Each agent has a single responsibility and a specific position in the pipeline. No agent tries to do everything.

The key architectural insight: **planning before writing is not enough**. A real writing workflow has seven distinct stages — topic classification, research, planning, parallel writing, merging, fact verification, and SEO optimisation. DeepWrite has a dedicated node for each.

## User Flow

1. User enters a topic and clicks **Generate Article**
2. The **Router** classifies the topic — does it need live web research (open book) or can the LLM write from training knowledge (closed book)?
3. The **Memory** node retrieves past articles from a local ChromaDB vector store to inject the user's writing style into every section
4. If research is needed, the **Research** node queries Tavily and returns 4–5 grounded evidence items
5. The **Orchestrator** produces a structured writing plan: title, audience, tone, 4–6 sections with goals, bullets, and word targets
6. **Parallel Workers** write each section simultaneously — each one is then scored by a **Critic agent** on accuracy, depth, clarity, and grounding. Sections scoring below 6.5/10 are sent back for revision
7. The **Reducer** merges all sections into a single coherent article
8. The **Fact-Checker** extracts factual claims and cross-references each against the evidence pack, returning per-claim verdicts with confidence scores
9. The **SEO Audit** scores the article 0–100, generates a meta description, suggests keywords, and flags issues by severity
10. The user sees the finished article with live metric cards, can edit via the **AI Editor** (natural language chat), download as Markdown or styled HTML, and reload any past run from the draft history

## LLM Components

- **Router** — structured output (`RouterDecision`) classifying topic as `closed_book`, `hybrid`, or `open_book` and generating targeted search queries
- **Orchestrator** — structured output (`Plan` + `Task[]`) producing a full section-by-section writing plan with grounding policy per task
- **Worker agents** — parallel LLM calls writing individual sections, each with a distinct system prompt enforcing grounding rules and style context from memory
- **Critic agent** — structured output (`CriticScore`) scoring four dimensions (accuracy, depth, clarity, grounding) and generating specific revision feedback when quality is below threshold
- **Fact-Checker** — structured output (`FactCheckReport`) extracting and verifying 5–10 factual claims against the evidence pack, with per-claim confidence scores and supporting URLs
- **SEO Auditor** — structured output (`SEOReport`) scoring readability, keyword density, heading structure, and generating a publication-ready meta description
- **AI Editor** — conversational LLM with full article context, conversation memory (last 10 turns), and a structured response protocol distinguishing EDIT, QUESTION, and SUGGESTION modes
- **Evidence fallback** — when Tavily is unavailable, an LLM call synthesises representative evidence items so the pipeline never fails silently

## Tools

- **LangGraph** — state machine managing the full 7-node pipeline with fan-out/fan-in for parallel workers and subgraph composition for the reducer
- **Groq (Llama 3.3 70B)** — primary LLM for all nodes; chosen for free-tier availability and speed (~5x faster than GPT-4o on comparable tasks)
- **Tavily API** — real-time web search for research-mode topics with graceful fallback
- **ChromaDB** — local persistent vector store for writer memory (stores and retrieves past articles by semantic similarity)
- **sentence-transformers (all-MiniLM-L6-v2)** — embedding model for ChromaDB; runs locally, no API key required
- **Streamlit** — frontend dashboard with live pipeline tracker, tabbed results, AI Editor chat interface, and SQLite-backed draft history
- **SQLite** — persistent draft history storing every article run with metadata (words, SEO score, mode)
- **python-dotenv** — environment management for API keys

## Try It

[View on GitHub](https://github.com/iMeet07/deepwrite)

**Setup:**
```bash
pip install -r requirements.txt
# Add to .env: XAI_API_KEY, TAVILY_API_KEY
streamlit run bwa_frontend.py
```
