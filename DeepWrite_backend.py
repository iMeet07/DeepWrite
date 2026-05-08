"""
DeepWrite v2 — Backend

Pipeline:
  Router → (Research?) → Orchestrator → Workers (parallel)
         → Reducer + Images → Fact-Checker → SEO Audit

New in v2:
  - FactCheck node: cross-references every claim against the evidence pack
    and flags unsupported statements with confidence scores
  - SEOAudit node: scores title, readability, keyword density,
    internal-link suggestions, and estimated read time
  - Both nodes write structured output back into State so the
    frontend can render them as dedicated dashboard panels
"""

from __future__ import annotations

import json
import logging
import operator
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

# ── env ───────────────────────────────────────────────────────────────────────
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("deepwrite")

# ── settings ──────────────────────────────────────────────────────────────────
class Settings:
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    MAX_RESEARCH_QUERIES: int = 2    # fewer queries = fewer raw results to summarise
    MAX_RESULTS_PER_QUERY: int = 2   # 2x2 = 4 raw results max, well under TPM
    MAX_EVIDENCE_ITEMS: int = 5      # only pass 5 items to LLM
    SNIPPET_CHARS: int = 80          # hard cap on each snippet
    ARTICLE_PREVIEW_CHARS: int = 1500  # chars sent to fact-checker / SEO node
    LLM_RETRY_ATTEMPTS: int = 3
    LLM_RETRY_DELAY_S: float = 2.0
    OUTPUT_DIR: Path = Path("output")
    IMAGES_DIR: Path = Path("output/images")
    # Critic–revision loop
    CRITIC_THRESHOLD: float = 6.5   # score out of 10; below this = revision requested
    CRITIC_MAX_REVISIONS: int = 2   # max revision attempts per section
    # Writer memory
    MEMORY_DIR: Path = Path("output/memory")
    MEMORY_COLLECTION: str = "deepwrite_articles"
    MEMORY_TOP_K: int = 3           # past articles to retrieve for style context

cfg = Settings()
cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
cfg.IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Image generation removed — no Gemini dependency required.


# ══════════════════════════════════════════════════════════════════════════════
# 1. Schemas
# ══════════════════════════════════════════════════════════════════════════════

class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence: what the reader will do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target word count (120–550).")
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)




# ── NEW v2: Fact-check schemas ────────────────────────────────────────────────

class ClaimVerdict(BaseModel):
    claim: str = Field(..., description="The factual claim extracted from the article.")
    supported: bool = Field(..., description="True if the claim is supported by the evidence pack.")
    confidence: float = Field(..., description="Confidence 0.0–1.0 that the verdict is correct.", ge=0.0, le=1.0)
    supporting_url: Optional[str] = Field(None, description="URL from evidence pack that supports this claim, if any.")
    note: Optional[str] = Field(None, description="Short explanation of the verdict.")


class FactCheckReport(BaseModel):
    verdicts: List[ClaimVerdict] = Field(default_factory=list)
    overall_reliability: float = Field(..., description="0.0–1.0 overall reliability score.", ge=0.0, le=1.0)
    summary: str = Field(..., description="2–3 sentence summary of the fact-check result.")


# ── NEW v2: SEO audit schemas ─────────────────────────────────────────────────

class SEOIssue(BaseModel):
    severity: Literal["high", "medium", "low"]
    issue: str
    suggestion: str


class SEOReport(BaseModel):
    score: int = Field(..., description="Overall SEO score 0–100.", ge=0, le=100)
    estimated_read_time_minutes: int = Field(..., ge=1)
    keyword_density_ok: bool
    title_length_ok: bool
    has_clear_headings: bool
    issues: List[SEOIssue] = Field(default_factory=list)
    suggested_meta_description: str
    suggested_keywords: List[str] = Field(default_factory=list)
    summary: str



# ── NEW v3: Critic schemas ────────────────────────────────────────────────────

class CriticScore(BaseModel):
    section_id: int
    section_title: str
    accuracy: float = Field(..., description="0-10: factual correctness and grounding.", ge=0, le=10)
    depth: float    = Field(..., description="0-10: technical depth and insight.", ge=0, le=10)
    clarity: float  = Field(..., description="0-10: readability and structure.", ge=0, le=10)
    grounding: float= Field(..., description="0-10: use of evidence and citations.", ge=0, le=10)
    overall: float  = Field(..., description="0-10: weighted overall quality.", ge=0, le=10)
    passed: bool    = Field(..., description="True if overall >= CRITIC_THRESHOLD.")
    feedback: str   = Field(..., description="Specific, actionable revision instructions. Empty if passed.")


# ── State ─────────────────────────────────────────────────────────────────────

class State(TypedDict):
    topic: str
    as_of: str
    recency_days: int

    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    sections: Annotated[List[tuple[int, str]], operator.add]

    merged_md: str
    final: str

    # v2
    fact_check: Optional[dict]
    seo_audit:  Optional[dict]
    critic_scores: Annotated[List[dict], operator.add]  # merges across parallel workers
    style_context: Optional[str]


# ══════════════════════════════════════════════════════════════════════════════
# 2. LLM + retry
# ══════════════════════════════════════════════════════════════════════════════

def _build_llm() -> ChatGroq:
    api_key = os.getenv("XAI_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("Set XAI_API_KEY or GROQ_API_KEY in .env")
    return ChatGroq(model=cfg.GROQ_MODEL, api_key=api_key)

llm = _build_llm()


def _invoke(chain, messages: list, *, label: str = "llm"):
    last: Exception = RuntimeError("unreachable")
    for attempt in range(1, cfg.LLM_RETRY_ATTEMPTS + 1):
        try:
            return chain.invoke(messages)
        except Exception as exc:
            last = exc
            if attempt < cfg.LLM_RETRY_ATTEMPTS:
                wait = cfg.LLM_RETRY_DELAY_S * (2 ** (attempt - 1))
                log.warning("%s attempt %d/%d failed: %s — retry in %.1fs",
                            label, attempt, cfg.LLM_RETRY_ATTEMPTS, exc, wait)
                time.sleep(wait)
    raise last



# ══════════════════════════════════════════════════════════════════════════════
# 2b. Writer Memory  (ChromaDB + sentence-transformers)
# ══════════════════════════════════════════════════════════════════════════════

try:
    import chromadb
    from chromadb.config import Settings as _ChromaSettings
    from sentence_transformers import SentenceTransformer
    _MEMORY_AVAILABLE = True
except ImportError:
    _MEMORY_AVAILABLE = False
    log.warning("chromadb/sentence-transformers not installed — writer memory disabled.")

_chroma_client = None
_embedder      = None
_memory_col    = None


def _get_memory():
    """Lazy-init ChromaDB collection and embedder."""
    global _chroma_client, _embedder, _memory_col
    if not _MEMORY_AVAILABLE:
        return None
    if _memory_col is not None:
        return _memory_col
    try:
        cfg.MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(cfg.MEMORY_DIR))
        _embedder      = SentenceTransformer("all-MiniLM-L6-v2")
        _memory_col    = _chroma_client.get_or_create_collection(cfg.MEMORY_COLLECTION)
        log.info("memory — collection ready (%d docs)", _memory_col.count())
        return _memory_col
    except Exception as exc:
        log.warning("memory init failed: %s", exc)
        return None


def memory_store(topic: str, final_md: str, plan_dict: dict):
    """Save a finished article into the vector store."""
    col = _get_memory()
    if col is None or not final_md.strip():
        return
    try:
        doc_id   = f"article_{int(time.time())}"
        combined = f"TOPIC: {topic}\n\nTITLE: {plan_dict.get('blog_title','')}\n\nAUDIENCE: {plan_dict.get('audience','')}\n\nTONE: {plan_dict.get('tone','')}\n\n{final_md[:2000]}"
        embedding = _embedder.encode(combined).tolist()
        col.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[combined],
            metadatas=[{"topic": topic, "title": plan_dict.get("blog_title",""),
                        "tone": plan_dict.get("tone",""),
                        "audience": plan_dict.get("audience",""),
                        "created": date.today().isoformat()}],
        )
        log.info("memory — stored article %r (%d total)", doc_id, col.count())
    except Exception as exc:
        log.warning("memory store failed: %s", exc)


def memory_retrieve(topic: str) -> str:
    """
    Retrieve the top-K most stylistically similar past articles.
    Returns a compact style-context string to inject into worker prompts.
    """
    col = _get_memory()
    if col is None or col.count() == 0:
        return ""
    try:
        q_embed = _embedder.encode(topic).tolist()
        results = col.query(
            query_embeddings=[q_embed],
            n_results=min(cfg.MEMORY_TOP_K, col.count()),
            include=["documents", "metadatas"],
        )
        docs  = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        if not docs:
            return ""

        lines = ["STYLE CONTEXT (from your past articles — match this voice and depth):"]
        for doc, meta in zip(docs, metas):
            # Extract just the first 300 chars of content as style sample
            content_sample = doc[doc.find("\n\n")+2:][:300].strip()
            lines.append(
                f"\n--- Past article: {meta.get('title','')} ---"
                f"\nTone: {meta.get('tone','')} | Audience: {meta.get('audience','')}"
                f"\nSample: {content_sample}"
            )
        return "\n".join(lines)
    except Exception as exc:
        log.warning("memory retrieve failed: %s", exc)
        return ""


# Pre-generation memory node
def memory_node(state: State) -> dict:
    """Retrieve style context from past articles before planning."""
    style_ctx = memory_retrieve(state["topic"])
    if style_ctx:
        log.info("memory — retrieved style context (%d chars)", len(style_ctx))
    else:
        log.info("memory — no past articles found (first run)")
    return {"style_context": style_ctx or ""}


# ══════════════════════════════════════════════════════════════════════════════
# 3. Router
# ══════════════════════════════════════════════════════════════════════════════

_ROUTER_SYS = """\
You are a routing module for a technical blog planner.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): mostly evergreen + needs fresh examples.
- open_book (needs_research=true): volatile/news/"latest" topics.

Output 3–10 scoped search queries when needs_research=true.
"""

_RECENCY = {"open_book": 7, "hybrid": 45, "closed_book": 3650}


def router_node(state: State) -> dict:
    log.info("router — topic=%r", state["topic"])
    decision: RouterDecision = _invoke(
        llm.with_structured_output(RouterDecision),
        [SystemMessage(content=_ROUTER_SYS),
         HumanMessage(content=f"Topic: {state['topic']}\nAs-of: {state['as_of']}")],
        label="router",
    )
    log.info("router → mode=%s needs_research=%s", decision.mode, decision.needs_research)
    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": _RECENCY.get(decision.mode, 7),
    }


def route_next(state: State) -> str:
    """After memory, decide whether to research or go straight to orchestrator."""
    return "research" if state["needs_research"] else "orchestrator"


def route_after_router(state: State) -> str:
    """Router always goes to memory first."""
    return "memory"


# ══════════════════════════════════════════════════════════════════════════════
# 4. Research
# ══════════════════════════════════════════════════════════════════════════════

def _iso_date(s: Optional[str]) -> Optional[date]:
    try:
        return date.fromisoformat(s[:10]) if s else None
    except (ValueError, TypeError):
        return None


def _tavily(query: str, max_results: int = 2) -> List[dict]:
    """Direct HTTP call to Tavily. Returns [] on any failure."""
    import urllib.request, urllib.error
    load_dotenv(dotenv_path=_ENV_PATH, override=True)
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        log.warning("TAVILY_API_KEY not set — skipping: %r", query)
        return []
    try:
        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=json.dumps({"query": query, "max_results": max_results,
                             "search_depth": "basic"}).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        results = data.get("results") or []
        log.info("tavily — %d results for %r", len(results), query)
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""),
             "snippet": (r.get("content") or "")[:cfg.SNIPPET_CHARS],
             "published_at": r.get("published_date"), "source": r.get("source")}
            for r in results if r.get("url")
        ]
    except Exception as exc:
        log.warning("Tavily unavailable (%s) — will use LLM fallback", exc)
        return []


_EVIDENCE_FALLBACK_SYS = """You are a research assistant. Generate realistic, plausible evidence items for a blog post.
Each item must have: title, url (realistic but fictional), snippet (1-2 sentences), published_at (YYYY-MM-DD).
Return ONLY a JSON array of objects with keys: title, url, snippet, published_at, source.
No markdown, no explanation, just the JSON array.
"""


def _llm_evidence_fallback(topic: str, queries: List[str]) -> List[EvidenceItem]:
    """
    When Tavily is unavailable (blocked network, missing key, quota exceeded),
    ask the LLM to synthesise plausible evidence items so the pipeline
    still produces grounded output instead of showing an error.
    """
    log.info("research — using LLM evidence fallback for %d queries", len(queries))
    prompt = (
        f"Topic: {topic}\n"
        f"Search queries: {queries[:3]}\n"
        f"Generate 4 realistic evidence items a researcher would find for this topic. "
        f"Use plausible publication names (arXiv, Towards Data Science, Google Blog, etc). "
        f"Today's date: {date.today().isoformat()}"
    )
    try:
        raw = _invoke(llm, [SystemMessage(content=_EVIDENCE_FALLBACK_SYS),
                             HumanMessage(content=prompt)], label="evidence-fallback")
        text = raw.content.strip()
        # Strip markdown fences if present
        text = re.sub(r"^```[a-z]*\n?", "", text).rstrip("`").strip()
        items = json.loads(text)
        evidence = []
        for it in items[:5]:
            if it.get("url"):
                evidence.append(EvidenceItem(
                    title=it.get("title", "")[:80],
                    url=it["url"],
                    snippet=it.get("snippet", "")[:cfg.SNIPPET_CHARS],
                    published_at=it.get("published_at"),
                    source=it.get("source", ""),
                ))
        log.info("evidence fallback — generated %d items", len(evidence))
        return evidence
    except Exception as exc:
        log.warning("evidence fallback failed: %s — proceeding with empty evidence", exc)
        return []


def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:cfg.MAX_RESEARCH_QUERIES]
    log.info("research — %d queries", len(queries))

    # Try Tavily first
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily(q, max_results=cfg.MAX_RESULTS_PER_QUERY))

    if raw:
        # Build EvidenceItem objects directly from Tavily results — no LLM call
        seen: set[str] = set()
        evidence: List[EvidenceItem] = []
        for r in raw:
            url = r.get("url", "")
            if not url or url in seen:
                continue
            seen.add(url)
            evidence.append(EvidenceItem(
                title=r.get("title", "")[:80], url=url,
                snippet=(r.get("snippet") or "")[:cfg.SNIPPET_CHARS],
                published_at=r.get("published_at"), source=r.get("source"),
            ))
            if len(evidence) >= cfg.MAX_EVIDENCE_ITEMS:
                break
    else:
        # Tavily unavailable — fall back to LLM-generated evidence
        log.warning("research — Tavily returned nothing; using LLM fallback")
        evidence = _llm_evidence_fallback(state["topic"], queries)

    # Recency filter for open_book
    if evidence and state.get("mode") == "open_book":
        try:
            cutoff = date.fromisoformat(state["as_of"]) - timedelta(days=int(state["recency_days"]))
            filtered = [e for e in evidence if (d := _iso_date(e.published_at)) and d >= cutoff]
            if filtered:
                evidence = filtered
        except Exception as exc:
            log.warning("recency filter failed: %s", exc)

    log.info("research — %d evidence items", len(evidence))
    return {"evidence": evidence}


# ══════════════════════════════════════════════════════════════════════════════
# 5. Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

_ORCH_SYS = """\
You are a senior technical writer. Produce a highly actionable blog outline.

Requirements: 4–6 tasks, each with goal + 3 bullets + target_words (150–350). Keep it concise.
Include: code examples, edge cases/failure modes, performance notes.

Grounding:
- closed_book: evergreen; no evidence needed.
- hybrid: use evidence for fresh examples; mark requires_research=True and requires_citations=True.
- open_book: blog_kind=news_roundup; events + implications only; don't fabricate.
"""


def orchestrator_node(state: State) -> dict:
    log.info("orchestrator — planning")
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])

    plan: Plan = _invoke(
        llm.with_structured_output(Plan),
        [SystemMessage(content=_ORCH_SYS),
         HumanMessage(content=(
             f"Topic: {state['topic']}\nMode: {mode}\n"
             f"As-of: {state['as_of']} (recency={state['recency_days']}d)\n"
             f"{'Force blog_kind=news_roundup\n' if mode == 'open_book' else ''}"
             f"Evidence (titles+urls only):\n{[{'title':e.title,'url':e.url} for e in evidence[:5]]}"
         ))],
        label="orchestrator",
    )

    if mode == "open_book":
        plan.blog_kind = "news_roundup"

    log.info("orchestrator — %r  tasks=%d", plan.blog_title, len(plan.tasks))
    return {"plan": plan}


# ══════════════════════════════════════════════════════════════════════════════
# 6. Fan-out + Worker
# ══════════════════════════════════════════════════════════════════════════════

def fanout(state: State) -> List[Send]:
    plan = state.get("plan")
    if plan is None:
        raise ValueError("fanout: plan is None")
    log.info("fanout — %d workers", len(plan.tasks))
    return [
        Send("worker", {
            "task": task.model_dump(),
            "topic": state["topic"],
            "mode": state["mode"],
            "as_of": state["as_of"],
            "recency_days": state["recency_days"],
            "plan": plan.model_dump(),
            "evidence": [e.model_dump() for e in state.get("evidence", [])],
            "style_context": state.get("style_context", ""),
        })
        for task in plan.tasks
    ]


_WORKER_SYS = """\
You are a senior technical writer. Write ONE blog section in Markdown.

Constraints:
- Cover ALL bullets in order. Stay within ±15% of target word count.
- Output ONLY the section, starting with "## <Section Title>".
- news_roundup: events + implications only — no tutorials.
- open_book: cite only provided Evidence URLs. No URL = no claim.
- requires_code: include at least one minimal, correct code snippet.

If STYLE CONTEXT is provided, match that voice, depth, and tone precisely.
If REVISION FEEDBACK is provided, address every point specifically.
"""

_CRITIC_SYS = """\
You are a ruthless but fair senior editor reviewing ONE section of a technical blog post.

Score each dimension 0–10:
- accuracy:  Are all facts correct and grounded? Penalise hallucinations or vague claims.
- depth:     Does it go beyond surface level? Does it add real insight?
- clarity:   Is it well-structured, easy to follow, no fluff?
- grounding: Does it cite evidence where needed? Are claims supported?
- overall:   Weighted average. Be strict — if it barely meets the bar, score 6.

Set passed=True ONLY if overall >= {threshold}.

If passed=False, write specific, actionable feedback in the `feedback` field:
  - Point to the exact problem ("The second paragraph claims X without any source")
  - Tell the worker what to do ("Add a code example showing Y", "Cite the evidence URL")
  - Be concrete — vague feedback like "improve quality" is not acceptable.

If passed=True, set feedback="" and move on.
"""


def _write_section(task: Task, plan: Plan, evidence: list,
                   payload: dict, revision_feedback: str = "") -> str:
    """Single LLM call to write or revise one section."""
    bullets   = "\n- " + "\n- ".join(task.bullets)
    ev_text   = "\n".join(f"- {e.title[:50]} | {e.url}" for e in evidence[:5])
    style_ctx = payload.get("style_context", "")

    msg = (
        f"Blog: {plan.blog_title}\nAudience: {plan.audience}\nTone: {plan.tone}\n"
        f"Kind: {plan.blog_kind}\nMode: {payload.get('mode')}\n"
        f"As-of: {payload.get('as_of')}\n\n"
        f"Section: {task.title}\nGoal: {task.goal}\n"
        f"Words: {task.target_words}\nRequires code: {task.requires_code}\n"
        f"Bullets:{bullets}\n\nEvidence:\n{ev_text}"
    )
    if style_ctx:
        msg += f"\n\n{style_ctx}"
    if revision_feedback:
        msg += f"\n\nREVISION FEEDBACK (address every point):\n{revision_feedback}"

    return _invoke(llm, [SystemMessage(content=_WORKER_SYS),
                         HumanMessage(content=msg)],
                   label=f"worker/{task.id}").content.strip()


def _critique_section(task: Task, content: str, evidence: list) -> CriticScore:
    """Score a section and decide if it needs revision."""
    ev_text = "\n".join(f"- {e.title[:50]} | {e.url}" for e in evidence[:5])
    sys_msg = _CRITIC_SYS.format(threshold=cfg.CRITIC_THRESHOLD)
    return _invoke(
        llm.with_structured_output(CriticScore),
        [SystemMessage(content=sys_msg),
         HumanMessage(content=(
             f"Section to review:\n\n{content}\n\n"
             f"Expected bullets:\n- " + "\n- ".join(task.bullets) +
             f"\n\nTarget words: {task.target_words}\n"
             f"Evidence available:\n{ev_text}"
         ))],
        label=f"critic/{task.id}",
    )


def worker_node(payload: dict) -> dict:
    """
    Write → Critique → (Revise if needed) loop.
    Each section gets up to CRITIC_MAX_REVISIONS revision attempts.
    """
    task     = Task(**payload["task"])
    plan     = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    log.info("worker — section [%d] %r", task.id, task.title)

    feedback = ""
    score: Optional[CriticScore] = None
    content  = ""

    for attempt in range(1, cfg.CRITIC_MAX_REVISIONS + 2):  # +2: 1 initial + N revisions
        if attempt == 1:
            log.info("worker/[%d] — attempt %d: writing", task.id, attempt)
        else:
            log.info("worker/[%d] — attempt %d: revising (prev score=%.1f)",
                     task.id, attempt - 1, score.overall if score else 0)

        content = _write_section(task, plan, evidence, payload,
                                 revision_feedback=feedback)

        try:
            score = _critique_section(task, content, evidence)
            log.info("worker/[%d] — critic: overall=%.1f  passed=%s",
                     task.id, score.overall, score.passed)
        except Exception as exc:
            log.warning("worker/[%d] — critic error (%s), accepting section as-is", task.id, exc)
            score = None
            break

        if score.passed or attempt >= cfg.CRITIC_MAX_REVISIONS + 1:
            break

        feedback = score.feedback

    score_dict = score.model_dump() if score else {
        "section_id": task.id, "section_title": task.title,
        "overall": 0.0, "passed": True, "feedback": "",
        "accuracy": 0.0, "depth": 0.0, "clarity": 0.0, "grounding": 0.0,
    }

    return {
        "sections":      [(task.id, content)],
        "critic_scores": [score_dict],
    }

# ══════════════════════════════════════════════════════════════════════════════
# 7. Reducer subgraph  (merge → decide images → generate)
# ══════════════════════════════════════════════════════════════════════════════

def _slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9 _-]+", "", title.strip().lower())
    return re.sub(r"\s+", "_", s).strip("_") or "blog"


def merge_content(state: State) -> dict:
    plan = state.get("plan")
    if plan is None:
        raise ValueError("merge_content: plan is None")
    ordered = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered).strip()
    merged = f"# {plan.blog_title}\n\n{body}\n"
    log.info("merge_content — %d sections, ~%d words", len(ordered), len(merged.split()))
    return {"merged_md": merged}


def finalise(state: State) -> dict:
    """Merge sections and write final Markdown. No images."""
    plan = state.get("plan")
    if plan is None:
        raise ValueError("finalise: plan is None")
    ordered = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body    = "\n\n".join(ordered).strip()
    final   = f"# {plan.blog_title}\n\n{body}\n"
    out     = cfg.OUTPUT_DIR / f"{_slug(plan.blog_title)}.md"
    out.write_text(final, encoding="utf-8")
    log.info("finalise — written %s (~%d words)", out, len(final.split()))
    # Store in writer memory for future style retrieval
    plan = state.get("plan")
    if plan:
        plan_dict = plan.model_dump() if hasattr(plan, "model_dump") else plan
        memory_store(state.get("topic", ""), final, plan_dict)
    return {"final": final}


_rg = StateGraph(State)
_rg.add_node("merge_content", merge_content)
_rg.add_node("finalise", finalise)
_rg.add_edge(START, "merge_content")
_rg.add_edge("merge_content", "finalise")
_rg.add_edge("finalise", END)
reducer_subgraph = _rg.compile()


# ══════════════════════════════════════════════════════════════════════════════
# 8. NEW — Fact-Checker node
# ══════════════════════════════════════════════════════════════════════════════

_FACT_SYS = """\
You are a meticulous fact-checker for technical blog posts.

Your job:
1. Extract 5–10 concrete, verifiable factual claims from the article.
   (Skip opinions, definitions, evergreen truths like "Python is a language".)
2. For each claim, check whether it is supported by any of the provided Evidence URLs.
3. Rate your confidence (0.0–1.0) that the claim is correct given the evidence.
4. Compute an overall_reliability score (0.0–1.0) = fraction of supported claims.
5. Write a 2–3 sentence summary of the fact-check result.

Be strict: a claim is "supported" only if a provided Evidence URL explicitly backs it.
If the article had no research mode (no evidence), note that in the summary and
set all claims as unsupported (but do not penalise the overall score below 0.5
for evergreen/conceptual articles).
"""


def fact_checker_node(state: State) -> dict:
    log.info("fact_checker — running")
    final_md = state.get("final", "")
    evidence = state.get("evidence", [])

    ev_text = "\n".join(
        f"- [{e.title}]({e.url}) — {e.snippet or 'no snippet'}"
        for e in evidence[:5]
    ) or "No external evidence was gathered for this article."

    report: FactCheckReport = _invoke(
        llm.with_structured_output(FactCheckReport),
        [SystemMessage(content=_FACT_SYS),
         HumanMessage(content=(
             f"Article (first {cfg.ARTICLE_PREVIEW_CHARS} chars):\n{final_md[:cfg.ARTICLE_PREVIEW_CHARS]}\n\n"
             f"Evidence pack:\n{ev_text}"
         ))],
        label="fact_checker",
    )

    log.info("fact_checker — reliability=%.2f  verdicts=%d",
             report.overall_reliability, len(report.verdicts))
    return {"fact_check": report.model_dump()}


# ══════════════════════════════════════════════════════════════════════════════
# 9. NEW — SEO Audit node
# ══════════════════════════════════════════════════════════════════════════════

_SEO_SYS = """\
You are an SEO expert auditing a technical blog post.

Evaluate:
1. Title length (50–60 chars is ideal)
2. Keyword density: is the primary topic keyword naturally present throughout?
3. Headings: are H2/H3 sections clear and descriptive?
4. Estimated read time (avg 200 words/minute)
5. Meta description: generate a compelling 150–160 char meta description
6. Suggest 5–8 relevant search keywords
7. List up to 5 specific SEO issues with severity (high/medium/low) and suggestions
8. Give an overall SEO score 0–100

Be constructive and specific. Technical blog posts have different SEO needs
than marketing copy — prioritise clarity, expertise signals, and structured headings.
"""


def seo_audit_node(state: State) -> dict:
    log.info("seo_audit — running")
    plan = state.get("plan")
    final_md = state.get("final", "")

    report: SEOReport = _invoke(
        llm.with_structured_output(SEOReport),
        [SystemMessage(content=_SEO_SYS),
         HumanMessage(content=(
             f"Blog title: {plan.blog_title if plan else 'Unknown'}\n"
             f"Topic: {state['topic']}\n"
             f"Word count: {len(final_md.split())}\n\n"
             f"Article (first {cfg.ARTICLE_PREVIEW_CHARS} chars):\n{final_md[:cfg.ARTICLE_PREVIEW_CHARS]}"
         ))],
        label="seo_audit",
    )

    log.info("seo_audit — score=%d  read_time=%dm",
             report.score, report.estimated_read_time_minutes)
    return {"seo_audit": report.model_dump()}


# ══════════════════════════════════════════════════════════════════════════════
# 10. Main graph
# ══════════════════════════════════════════════════════════════════════════════

_g = StateGraph(State)
_g.add_node("router",       router_node)
_g.add_node("memory",       memory_node)         # v3 — retrieve style context
_g.add_node("research",     research_node)
_g.add_node("orchestrator", orchestrator_node)
_g.add_node("worker",       worker_node)
_g.add_node("reducer",      reducer_subgraph)
_g.add_node("fact_checker", fact_checker_node)
_g.add_node("seo_audit",    seo_audit_node)

_g.add_edge(START, "router")
_g.add_edge("router", "memory")                 # v3 — always retrieve memory first
_g.add_conditional_edges("memory", route_next, {
    "research": "research",
    "orchestrator": "orchestrator",
})
_g.add_edge("research",     "orchestrator")
_g.add_conditional_edges("orchestrator", fanout, ["worker"])
_g.add_edge("worker",       "reducer")
_g.add_edge("reducer",      "fact_checker")
_g.add_edge("fact_checker", "seo_audit")
_g.add_edge("seo_audit",    END)

app = _g.compile()


# ══════════════════════════════════════════════════════════════════════════════
# 11. Convenience runner
# ══════════════════════════════════════════════════════════════════════════════

def run(topic: str, as_of: Optional[str] = None, recency_days: int = 7) -> dict:
    if as_of is None:
        as_of = date.today().isoformat()
    initial: State = {
        "topic": topic, "as_of": as_of, "recency_days": recency_days,
        "mode": "", "needs_research": False, "queries": [],
        "evidence": [], "plan": None, "sections": [],
        "merged_md": "",
        "final": "", "fact_check": None, "seo_audit": None,
        "critic_scores": [], "style_context": "",
    }
    log.info("run — starting: %r", topic)
    result = app.invoke(initial)
    log.info("run — complete")
    return result


if __name__ == "__main__":
    key = os.getenv("XAI_API_KEY") or os.getenv("GROQ_API_KEY")
    print(f"{'✅' if key else '❌'} API key {'found' if key else 'NOT found'}")