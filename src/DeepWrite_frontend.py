"""
DeepWrite v2 — Frontend FINAL
Fixes:
  - _extract_state now correctly handles ALL LangGraph wrapping patterns:
      top-level node:  {"seo_audit": {"seo_audit": {...}}}  → state["seo_audit"] = {...}
      subgraph node:   {"reducer":   {"final": "..."}}       → state["final"] = "..."
      flat update:     {"router":    {"mode": "open_book"}}  → state["mode"] = "open_book"
  - Research tab shows actual evidence when mode is open_book/hybrid
  - New vibrant UI — gradients, colored badges, breathing room, personality
"""

import json
import logging
import re
import sqlite3
import time
import traceback
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

st.set_page_config(page_title="DeepWrite AI", layout="wide",
                   page_icon="✍️", initial_sidebar_state="expanded")

log = logging.getLogger('deepwrite')
from bwa_backend import Settings, State, Plan, app, llm, seo_audit_node, fact_checker_node, memory_retrieve
from langchain_core.messages import HumanMessage, SystemMessage

cfg = Settings()

# ══════════════════════════════════════════════════════════════════════════════
# THEME  — vibrant, modern, alive
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*{box-sizing:border-box}
html,body,.stApp{font-family:'Inter',sans-serif!important;background:#070B12!important}
.stApp{color:#C9D1D9}

/* ── Sidebar ── */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0D1117 0%,#070B12 100%)!important;
  border-right:1px solid #1C2333!important}
[data-testid="stSidebar"] .stTextArea textarea{
  background:#0D1117!important;border:1px solid #1C2333!important;
  color:#C9D1D9!important;border-radius:10px!important}
[data-testid="stSidebar"] .stTextArea textarea:focus{border-color:#388BFD!important}

/* ── Logo ── */
.dw-logo{
  font-size:30px;font-weight:800;letter-spacing:-1.5px;
  background:linear-gradient(135deg,#388BFD 0%,#58A6FF 40%,#79C0FF 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  margin-bottom:1px;line-height:1}
.dw-sub{font-size:10px;color:#484F58;letter-spacing:.12em;text-transform:uppercase;
  font-weight:500}

/* ── Pipeline tracker ── */
.pipeline-track{
  display:flex;margin:18px 0 8px;
  background:#0D1117;border:1px solid #1C2333;
  border-radius:14px;overflow:hidden;gap:1px}
.p-node{
  flex:1;padding:12px 4px 10px;text-align:center;
  font-size:10px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;
  color:#484F58;background:#0D1117;transition:all .4s;position:relative}
.p-node .p-icon{font-size:18px;display:block;margin-bottom:4px}
.p-node .p-time{font-size:9px;color:#30363D;margin-top:3px;font-weight:400}
.p-node.done{
  background:linear-gradient(180deg,#0A2116 0%,#071A10 100%);
  color:#3FB950}
.p-node.done::after{
  content:'';position:absolute;bottom:0;left:10%;right:10%;
  height:2px;background:#3FB950;border-radius:2px 2px 0 0}
.p-node.skip{background:#110D00;color:#9E6A03}
.p-node.skip::after{
  content:'';position:absolute;bottom:0;left:10%;right:10%;
  height:2px;background:#9E6A03;border-radius:2px 2px 0 0}
.p-node.active{
  background:linear-gradient(180deg,#0D1F3C 0%,#091529 100%);
  color:#58A6FF;animation:nodepulse 1.6s ease-in-out infinite}
.p-node.active::after{
  content:'';position:absolute;bottom:0;left:10%;right:10%;
  height:2px;background:#388BFD;border-radius:2px 2px 0 0;
  animation:barslide 1.6s ease-in-out infinite}
@keyframes nodepulse{0%,100%{opacity:1}50%{opacity:.65}}
@keyframes barslide{0%,100%{opacity:1}50%{opacity:.4}}

/* ── Metric cards ── */
.metric-row{display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 18px}
.metric-card{
  flex:1;min-width:130px;padding:16px 18px 14px;
  background:#0D1117;border:1px solid #1C2333;border-radius:14px;
  position:relative;overflow:hidden;transition:border-color .2s}
.metric-card:hover{border-color:#388BFD33}
.metric-card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:2px}
.mc-words::before{background:linear-gradient(90deg,#388BFD,#79C0FF)}
.mc-seo::before{background:linear-gradient(90deg,#3FB950,#56D364)}
.mc-rel::before{background:linear-gradient(90deg,#D29922,#E3B341)}
.mc-mode::before{background:linear-gradient(90deg,#BC8CFF,#D2A8FF)}
.mc-type::before{background:linear-gradient(90deg,#FF7B72,#FFA198)}
.m-label{font-size:10px;color:#484F58;letter-spacing:.1em;text-transform:uppercase;
  font-weight:600;margin-bottom:6px}
.m-value{font-size:24px;font-weight:700;color:#E6EDF3;line-height:1}
.m-sub{font-size:11px;color:#30363D;margin-top:5px}

/* ── Gauge numbers ── */
.gauge-wrap{text-align:center;padding:24px 0 10px}
.gauge-num{font-size:60px;font-weight:800;line-height:1;
  background:var(--gc);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.gauge-lbl{font-size:12px;color:#484F58;margin-top:6px;text-transform:uppercase;
  letter-spacing:.08em}
.gc-green{--gc:linear-gradient(135deg,#3FB950,#56D364)}
.gc-amber{--gc:linear-gradient(135deg,#D29922,#E3B341)}
.gc-red{--gc:linear-gradient(135deg,#F85149,#FF7B72)}

/* ── Issue chips ── */
.chip{display:inline-flex;align-items:center;gap:5px;border-radius:6px;
  padding:4px 10px;font-size:11px;font-weight:600;margin:3px}
.chip-high{background:#2D1515;color:#FF7B72;border:1px solid #5E1D1D}
.chip-medium{background:#231A00;color:#E3B341;border:1px solid #4B3600}
.chip-low{background:#0A1F0A;color:#56D364;border:1px solid #1A4D1A}

/* ── Keyword badges ── */
.kw-badge{
  display:inline-block;padding:4px 12px;margin:3px;font-size:12px;
  border-radius:20px;font-weight:500;
  background:linear-gradient(135deg,#1C2850,#162040);
  color:#79C0FF;border:1px solid #1F3060}

/* ── Confidence bar ── */
.conf-wrap{display:flex;align-items:center;gap:10px;margin:8px 0}
.conf-bg{flex:1;height:8px;background:#1C2333;border-radius:4px;overflow:hidden}
.conf-fill{height:8px;border-radius:4px;transition:width .6s ease}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{
  gap:4px;border-bottom:1px solid #1C2333!important;
  background:transparent!important;padding-bottom:0}
.stTabs [data-baseweb="tab"]{
  background:#0D1117!important;border-radius:10px 10px 0 0!important;
  color:#484F58!important;padding:9px 16px!important;font-size:12px!important;
  font-weight:500!important;letter-spacing:.02em!important;
  border:1px solid #1C2333!important;border-bottom:none!important;
  transition:all .2s!important}
.stTabs [data-baseweb="tab"]:hover{color:#8B949E!important;background:#141B24!important}
.stTabs [aria-selected="true"]{
  background:linear-gradient(180deg,#141B24 0%,#0D1117 100%)!important;
  color:#E6EDF3!important;border-color:#388BFD!important;
  border-bottom:1px solid #0D1117!important}

/* ── Buttons ── */
.stButton>button{
  background:#141B24!important;color:#C9D1D9!important;
  border:1px solid #1C2333!important;border-radius:10px!important;
  font-size:13px!important;font-weight:500!important;
  transition:all .2s!important}
.stButton>button:hover{
  background:#1C2333!important;border-color:#388BFD!important;
  color:#79C0FF!important;transform:translateY(-1px)}

/* ── Download buttons ── */
[data-testid="stDownloadButton"]>button{
  background:linear-gradient(135deg,#161C28,#0D1117)!important;
  border:1px solid #1C2333!important;color:#C9D1D9!important;
  border-radius:10px!important;font-size:13px!important;transition:all .2s!important}
[data-testid="stDownloadButton"]>button:hover{
  border-color:#388BFD!important;color:#79C0FF!important}

/* ── Primary button override ── */
[data-testid="stSidebar"] .stButton>button[kind="primary"]{
  background:linear-gradient(135deg,#1A3A6E,#1158C7)!important;
  border:1px solid #388BFD!important;color:#fff!important;
  font-weight:600!important;font-size:14px!important;
  padding:12px!important;letter-spacing:.02em}
[data-testid="stSidebar"] .stButton>button[kind="primary"]:hover{
  background:linear-gradient(135deg,#1C4480,#1A6AE0)!important;
  transform:translateY(-1px);box-shadow:0 4px 20px #388BFD33!important}

/* ── Chat ── */
.chat-wrap{display:flex;flex-direction:column;gap:16px;padding:8px 0}
.cmsg{display:flex;gap:10px;align-items:flex-start}
.cmsg.user{flex-direction:row-reverse}
.cavatar{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:11px;font-weight:700;flex-shrink:0;margin-top:2px}
.av-ai{
  background:linear-gradient(135deg,#1C2850,#1A3A6E);
  color:#79C0FF;border:1px solid #1F4068}
.av-user{
  background:linear-gradient(135deg,#0A1F0A,#102A10);
  color:#56D364;border:1px solid #1A4D1A}
.cbubble{max-width:78%;padding:12px 16px;font-size:13px;line-height:1.65}
.cbubble.ai{
  background:linear-gradient(135deg,#0D1117,#111820);
  border:1px solid #1C2333;color:#C9D1D9;border-radius:4px 14px 14px 14px}
.cbubble.user{
  background:linear-gradient(135deg,#141C30,#0D1525);
  border:1px solid #1F3060;color:#A8CCFF;border-radius:14px 4px 14px 14px}
.cbubble code{
  background:#161B22;border:1px solid #30363D;border-radius:4px;
  padding:1px 6px;font-size:12px;color:#79C0FF}
.edit-badge{
  display:inline-block;font-size:10px;font-weight:600;letter-spacing:.05em;
  background:linear-gradient(135deg,#0A2116,#071A10);
  color:#3FB950;border:1px solid #1A4D1A;
  border-radius:6px;padding:3px 8px;margin-bottom:8px}
.chat-empty{
  text-align:center;padding:40px 20px;
  background:linear-gradient(180deg,#0D1117,#070B12);
  border:1px dashed #1C2333;border-radius:16px;margin:12px 0}
.chat-empty-title{font-size:14px;font-weight:500;color:#484F58;margin-bottom:8px}
.chat-empty-hint{font-size:12px;color:#30363D;line-height:1.7}

.undo-bar{
  display:flex;align-items:center;justify-content:space-between;
  background:linear-gradient(135deg,#0D1117,#111820);
  border:1px solid #1C2333;border-radius:10px;
  padding:10px 16px;font-size:12px;color:#484F58;margin-bottom:14px}

/* ── Section dividers ── */
.section-header{
  font-size:12px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
  color:#388BFD;margin:20px 0 12px;display:flex;align-items:center;gap:8px}
.section-header::after{content:'';flex:1;height:1px;background:#1C2333}

/* ── Welcome ── */
.welcome-wrap{
  background:linear-gradient(180deg,#0D1117 0%,#070B12 100%);
  border:1px solid #1C2333;border-radius:18px;padding:36px 40px;margin:12px 0}
.welcome-title{
  font-size:26px;font-weight:800;letter-spacing:-.5px;
  background:linear-gradient(135deg,#388BFD,#79C0FF);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  margin-bottom:10px}
.welcome-sub{font-size:14px;color:#484F58;line-height:1.65;margin-bottom:24px}
.fgrid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
.fitem{
  background:linear-gradient(135deg,#0D1117,#111820);
  border:1px solid #1C2333;border-radius:12px;padding:16px 18px;
  transition:border-color .2s}
.fitem:hover{border-color:#388BFD33}
.ft{font-size:13px;font-weight:600;margin-bottom:5px}
.fd{font-size:12px;color:#484F58;line-height:1.55}

/* ── Expanders ── */
[data-testid="stExpander"]{
  background:#0D1117!important;border:1px solid #1C2333!important;
  border-radius:10px!important}
[data-testid="stExpander"]:hover{border-color:#388BFD44!important}

/* ── Dataframe ── */
[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden}

/* ── Info/warning boxes ── */
.stAlert{border-radius:12px!important;border:none!important}

/* ── Code blocks ── */
.stCodeBlock{border-radius:10px!important}

/* ── Sidebar history rows ── */
.hist-meta{font-size:10px;color:#484F58;margin-top:2px}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _slug(s: str) -> str:
    return re.sub(r"\s+", "_",
           re.sub(r"[^a-z0-9 _-]+", "", s.strip().lower())).strip("_") or "article"


def _plan_attr(res: dict, attr: str, default=None):
    p = res.get("plan")
    if p is None: return default
    return p.get(attr, default) if isinstance(p, dict) else getattr(p, attr, default)


def _evidence_dicts(res: dict) -> List[dict]:
    return [i if isinstance(i, dict) else i.model_dump()
            for i in (res.get("evidence") or [])]


def _tasks_df(res: dict) -> Optional[pd.DataFrame]:
    p = res.get("plan")
    if not p: return None
    tasks = p.get("tasks", []) if isinstance(p, dict) else getattr(p, "tasks", [])
    rows = [t if isinstance(t, dict) else t.model_dump() for t in tasks]
    if not rows: return None
    df = pd.DataFrame(rows)
    cols = [c for c in ["id","title","target_words","requires_research","requires_code"]
            if c in df.columns]
    return df[cols] if cols else df


def _serialisable(res: dict) -> dict:
    out = {}
    for k, v in res.items():
        if hasattr(v, "model_dump"): out[k] = v.model_dump()
        elif isinstance(v, list):
            out[k] = [i.model_dump() if hasattr(i,"model_dump")
                      else (list(i) if isinstance(i,tuple) else i) for i in v]
        else: out[k] = v
    return out


def _extract_state(current: dict, event: dict) -> dict:
    """
    THE KEY FIX.

    LangGraph stream_mode="updates" emits events shaped like:
        {"<node_name>": <node_return_dict>}

    Where <node_return_dict> is EXACTLY what the node function returned.

    So:
        router        → {"router":       {"mode": "open_book", ...}}
        seo_audit     → {"seo_audit":    {"seo_audit": {...report...}}}
        fact_checker  → {"fact_checker": {"fact_check": {...report...}}}
        reducer (subgraph) → {"reducer": {"final": "...", "merged_md": "..."}}

    In ALL cases: iterate over the VALUES of the outer dict and merge them
    directly into current state. The node name key is just a routing label —
    the values are the actual state updates.
    """
    for _node_name, payload in event.items():
        if isinstance(payload, dict):
            current.update(payload)
        # non-dict payloads (rare) are ignored
    return current


# ══════════════════════════════════════════════════════════════════════════════
# DRAFT HISTORY
# ══════════════════════════════════════════════════════════════════════════════

DB_PATH = cfg.OUTPUT_DIR / "history.db"

def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS drafts(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created TEXT, topic TEXT, mode TEXT,
        words INTEGER, seo_score INTEGER, state_json TEXT)""")
    conn.commit()
    return conn

def save_draft(res: dict):
    try:
        seo = res.get("seo_audit") or {}
        with _db() as conn:
            conn.execute(
                "INSERT INTO drafts(created,topic,mode,words,seo_score,state_json) VALUES(?,?,?,?,?,?)",
                (datetime.now().isoformat(), res.get("topic",""),
                 res.get("mode",""), len((res.get("final") or "").split()),
                 seo.get("score") if isinstance(seo,dict) else None,
                 json.dumps(_serialisable(res))))
    except Exception: pass

def load_drafts() -> List[dict]:
    try:
        with _db() as conn:
            rows = conn.execute(
                "SELECT id,created,topic,mode,words,seo_score FROM drafts ORDER BY created DESC LIMIT 20"
            ).fetchall()
        return [{"id":r[0],"created":r[1],"topic":r[2],"mode":r[3],"words":r[4],"seo_score":r[5]}
                for r in rows]
    except Exception: return []

def load_draft_by_id(did: int) -> Optional[dict]:
    try:
        with _db() as conn:
            row = conn.execute("SELECT state_json FROM drafts WHERE id=?",(did,)).fetchone()
        return json.loads(row[0]) if row else None
    except Exception: return None

def delete_draft(did: int):
    try:
        with _db() as conn: conn.execute("DELETE FROM drafts WHERE id=?",(did,))
    except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════
# HTML EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def _inline_md(t):
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"\*(.+?)\*",     r"<em>\1</em>",          t)
    t = re.sub(r"`([^`]+)`",     r"<code>\1</code>",       t)
    return t

def _md_to_html(md):
    out, code_buf, code_lang = [], [], ""
    in_code = in_ul = False
    def close_ul():
        nonlocal in_ul
        if in_ul: out.append("</ul>"); in_ul = False
    for line in md.split("\n"):
        if line.startswith("```"):
            if not in_code:
                close_ul(); in_code = True
                code_lang = line[3:].strip() or "plaintext"; code_buf = []
            else:
                esc = "\n".join(code_buf).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                out.append(f'<pre><code class="language-{code_lang}">{esc}</code></pre>')
                in_code = False
            continue
        if in_code: code_buf.append(line); continue
        if   line.startswith("### "): close_ul(); out.append(f"<h3>{_inline_md(line[4:])}</h3>")
        elif line.startswith("## "):  close_ul(); out.append(f"<h2>{_inline_md(line[3:])}</h2>")
        elif line.startswith("# "):   close_ul()
        elif line.startswith(("- ","* ")):
            if not in_ul: out.append("<ul>"); in_ul = True
            out.append(f"  <li>{_inline_md(line[2:])}</li>")
        elif line.startswith("> "): close_ul(); out.append(f"<blockquote>{_inline_md(line[2:])}</blockquote>")
        elif line.strip() == "":    close_ul(); out.append("")
        else:                       close_ul(); out.append(f"<p>{_inline_md(line)}</p>")
    close_ul()
    return "\n".join(out)

def build_html(res: dict) -> str:
    title = _plan_attr(res,"blog_title") or res.get("topic","Untitled")
    kind  = (_plan_attr(res,"blog_kind","explainer") or "").replace("_"," ").title()
    mode  = (res.get("mode","") or "").replace("_"," ")
    final = res.get("final","")
    seo   = res.get("seo_audit") or {}
    if not isinstance(seo, dict): seo = {}
    wc    = len(final.split())
    rt    = seo.get("estimated_read_time_minutes") or max(1,round(wc/200))
    meta  = (seo.get("suggested_meta_description") or title)[:160]
    score = seo.get("score")
    kws   = seo.get("suggested_keywords",[]) or []
    if score is not None:
        cls = "g" if score>=70 else ("a" if score>=45 else "r")
        kw_html = "".join(f'<span class="kw">{k}</span>' for k in kws[:6])
        seo_card = (f'<div class="seo-card"><div class="seo-num {cls}">{score}</div>'
                    f'<div class="seo-lbl">SEO score</div><div class="kws">{kw_html}</div></div>')
    else:
        seo_card = ""
    tmpl = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="{meta}"><title>{title}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#070B12;color:#C9D1D9;font-family:Inter,sans-serif;line-height:1.75;font-size:16px}}
.wrap{{max-width:760px;margin:0 auto;padding:60px 24px 120px}}
header{{margin-bottom:48px;padding-bottom:32px;border-bottom:1px solid #1C2333}}
.meta{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px}}
.badge{{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;
  padding:4px 14px;border-radius:20px;
  background:linear-gradient(135deg,#1C2850,#162040);color:#79C0FF;border:1px solid #1F3060}}
h1{{font-size:2.1rem;font-weight:800;color:#E6EDF3;line-height:1.2;margin-bottom:14px}}
.summary{{font-size:16px;color:#484F58;line-height:1.65}}
h2{{font-size:1.4rem;font-weight:700;color:#E6EDF3;margin:2.5rem 0 1rem;
  padding-bottom:8px;border-bottom:1px solid #1C2333}}
h3{{font-size:1.1rem;font-weight:600;color:#C9D1D9;margin:1.8rem 0 .6rem}}
p{{margin-bottom:1rem}}a{{color:#58A6FF}}a:hover{{text-decoration:underline}}
ul,ol{{padding-left:1.5rem;margin-bottom:1rem}}li{{margin-bottom:.35rem}}
blockquote{{border-left:3px solid #388BFD;padding:10px 18px;background:#0D1117;
  border-radius:0 10px 10px 0;margin:1.5rem 0;color:#484F58}}
code{{font-family:'JetBrains Mono',monospace;font-size:.875em;background:#141B24;
  padding:2px 6px;border-radius:4px;color:#79C0FF}}
pre{{background:#0D1117;border:1px solid #1C2333;border-radius:12px;
  padding:20px;overflow-x:auto;margin:1.5rem 0}}
pre code{{background:none;padding:0;color:inherit}}
.seo-card{{float:right;width:185px;margin:0 0 24px 28px;background:#0D1117;
  border:1px solid #1C2333;border-radius:14px;padding:18px;text-align:center}}
.seo-num{{font-size:38px;font-weight:800}}
.seo-lbl{{font-size:11px;color:#484F58;margin-top:3px}}
.g{{color:#3FB950}}.a{{color:#D29922}}.r{{color:#F85149}}
.kws{{margin-top:10px;text-align:left}}
.kw{{font-size:11px;background:#1C2850;color:#79C0FF;border-radius:12px;
  padding:2px 8px;display:inline-block;margin:2px}}
.clearfix::after{{content:'';display:table;clear:both}}
footer{{margin-top:60px;padding-top:24px;border-top:1px solid #1C2333;
  font-size:12px;color:#30363D}}
</style></head><body><div class="wrap"><article>
<header><div class="meta">
  <span class="badge">{kind}</span>
  <span class="badge">{rt} min read</span>
  <span class="badge">{wc} words</span>
</div>
<h1>{title}</h1><p class="summary">{meta}</p></header>
<div class="clearfix">{seo_card}<div class="content">{body}</div></div>
<footer>Generated by <strong>DeepWrite</strong> · {dt} · Mode: {mode}</footer>
</article></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script>hljs.highlightAll();</script></body></html>"""
    return tmpl.format(title=title, kind=kind, rt=rt, wc=f"{wc:,}",
                       meta=meta, seo_card=seo_card, body=_md_to_html(final),
                       dt=date.today().strftime("%B %d, %Y"), mode=mode)


# ══════════════════════════════════════════════════════════════════════════════
# AI EDITOR
# ══════════════════════════════════════════════════════════════════════════════

_EDITOR_SYS = """\
You are an expert editor inside DeepWrite. You have the full article in context.

Decide the mode from the user's message:

1. EDIT — user wants to change the article.
   Write one brief line of explanation, then output the FULL updated article
   inside <article>...</article> tags. After </article> write:
   CHANGED: <one sentence describing exactly what changed>
   Keep edits targeted. Don't rewrite untouched sections.

2. QUESTION — user is asking about the article.
   Answer conversationally. Do NOT output <article> tags.

3. SUGGESTION — user wants ideas without committing.
   Give a numbered list. Do NOT output <article> tags.
"""

QUICK_ACTIONS = [
    ("✂️ Simplify intro",      "Simplify the introduction to be more beginner-friendly."),
    ("💻 Add code example",    "Add a practical code example to the most technical section."),
    ("🎯 Fix SEO issues",      "Improve keyword density and heading clarity based on SEO suggestions."),
    ("🎨 More casual tone",    "Rewrite the article in a more casual, conversational tone."),
    ("🔥 Stronger conclusion", "Rewrite the conclusion to be more impactful and memorable."),
    ("📏 Shorten by 20%",      "Shorten by about 20% — cut fluff, keep substance."),
    ("🔍 Review it",           "What are the 3 most impactful improvements I could make to this article?"),
    ("📊 Add a table",         "Add a Markdown comparison table to the most relevant section."),
]

def _run_editor(user_msg: str, article_md: str, history: List[dict]) -> dict:
    msgs = [
        SystemMessage(content=_EDITOR_SYS),
        HumanMessage(content=f"CURRENT ARTICLE:\n\n{article_md}\n\n---"),
    ]
    for turn in history[-10:]:
        if turn["role"] == "user":
            msgs.append(HumanMessage(content=turn["content"]))
        else:
            msgs.append(SystemMessage(content=f"[Your previous reply]: {turn['content'][:400]}"))
    msgs.append(HumanMessage(content=user_msg))
    raw = llm.invoke(msgs).content.strip()
    m = re.search(r"<article>(.*?)</article>", raw, re.DOTALL)
    if m:
        new_art = m.group(1).strip()
        cm = re.search(r"CHANGED:\s*(.+)", raw)
        summary = cm.group(1).strip() if cm else "Article updated."
        pre = raw[:raw.index("<article>")].strip()
        return {"reply": pre or summary, "new_article": new_art, "change_summary": summary}
    return {"reply": raw, "new_article": None, "change_summary": None}


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE TRACKER
# ══════════════════════════════════════════════════════════════════════════════

NODE_META = {
    "router":       ("🔀", "Router"),
    "memory":       ("🧠", "Memory"),
    "research":     ("🔍", "Research"),
    "orchestrator": ("📋", "Planner"),
    "worker":       ("✍️",  "Writers"),
    "reducer":      ("🔗", "Reducer"),
    "fact_checker": ("✅", "Fact Check"),
    "seo_audit":    ("📈", "SEO Audit"),
}

def render_pipeline(log_entries: List[dict]):
    done = {e["node"] for e in log_entries if e["status"]=="done"}
    skip = {e["node"] for e in log_entries if e["status"]=="skip"}
    cells = []
    for node,(icon,label) in NODE_META.items():
        elapsed = next((e["elapsed"] for e in log_entries if e["node"]==node), None)
        t_str = f"{elapsed:.1f}s" if elapsed else ""
        cls = "done" if node in done else "skip" if node in skip else ""
        cells.append(f'<div class="p-node {cls}"><span class="p-icon">{icon}</span>'
                     f'{label}<div class="p-time">{t_str}</div></div>')
    st.markdown(f'<div class="pipeline-track">{"".join(cells)}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

for k, v in [("result",None),("error",None),("pipeline_log",[]),
              ("editor_history",[]),("undo_stack",[])]:
    if k not in st.session_state: st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="dw-logo">DeepWrite.</div>', unsafe_allow_html=True)
    st.markdown('<div class="dw-sub">Multi-Agent Content Pipeline · v2</div>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    topic = st.text_area("Article Topic",
                         placeholder="How transformers changed NLP forever…", height=90)
    c1,c2 = st.columns(2)
    with c1: as_of = st.date_input("As-of date", value=date.today())
    with c2: days  = st.number_input("Recency (days)", value=7, min_value=1, max_value=3650)
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🚀  Generate Article", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**📂 Draft History**")
    history_rows = load_drafts()
    if not history_rows:
        st.caption("No saved drafts yet.")
    else:
        for row in history_rows[:6]:
            label = (row["topic"][:24]+"…") if len(row["topic"])>26 else row["topic"]
            words = f"{row['words']:,}w" if row["words"] else ""
            seo_s = f"· SEO {row['seo_score']}" if row["seo_score"] else ""
            ca,cb = st.columns([5,1])
            with ca:
                tip = f"{row['created'][:16].replace('T',' ')} · {words} {seo_s}"
                if st.button(f"↩ {label}", key=f"load_{row['id']}",
                             use_container_width=True, help=tip):
                    loaded = load_draft_by_id(row["id"])
                    if loaded:
                        st.session_state.result = loaded
                        st.session_state.error = None
                        st.session_state.pipeline_log = []
                        st.session_state.editor_history = []
                        st.session_state.undo_stack = []
                        st.rerun()
            with cb:
                if st.button("🗑", key=f"del_{row['id']}"):
                    delete_draft(row["id"]); st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# RUN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

tracker_slot = st.empty()

if run_btn:
    if not topic.strip():
        st.error("Please enter a topic first.")
    else:
        st.session_state.result = None
        st.session_state.error  = None
        st.session_state.pipeline_log  = []
        st.session_state.editor_history = []
        st.session_state.undo_stack    = []

        inputs: State = {
            "topic": topic, "as_of": as_of.isoformat(), "recency_days": int(days),
            "mode": "", "needs_research": False, "queries": [],
            "evidence": [], "plan": None, "sections": [],
            "merged_md": "", "final": "", "fact_check": None, "seo_audit": None,
            "critic_scores": [], "style_context": "",
        }

        current = dict(inputs)
        t_start: Dict[str,Optional[float]] = {n:None for n in NODE_META}
        n_workers = 0

        with st.status("⚙️ Pipeline running…", expanded=True) as status:
            try:
                for event in app.stream(inputs, stream_mode="updates"):
                    node = next(iter(event), "unknown")
                    now  = time.time()

                    # Merge state FIRST for every event
                    current = _extract_state(current, event)

                    if node == "worker":
                        n_workers += 1
                        plan = current.get("plan")
                        total = (len(plan.tasks) if plan and hasattr(plan,"tasks")
                                 else len(plan.get("tasks",[])) if isinstance(plan,dict)
                                 else "?")
                        status.write(f"✍️ Section {n_workers}/{total} written")
                        continue

                    if t_start.get(node) is None: t_start[node] = now
                    elapsed = round(now - (t_start[node] or now), 1)

                    if node in NODE_META:
                        st.session_state.pipeline_log.append(
                            {"node":node,"elapsed":elapsed,"status":"done"})
                        if node == "orchestrator":
                            done_set = {e["node"] for e in st.session_state.pipeline_log}
                            if "research" not in done_set:
                                st.session_state.pipeline_log.append(
                                    {"node":"research","elapsed":None,"status":"skip"})

                    status.write(f"✅ {NODE_META.get(node,('','Unknown'))[1]} done ({elapsed}s)")
                    with tracker_slot: render_pipeline(st.session_state.pipeline_log)

                if n_workers:
                    st.session_state.pipeline_log.append(
                        {"node":"worker","elapsed":None,"status":"done"})

                st.session_state.result = current
                save_draft(current)
                status.update(label="✨ Article ready!", state="complete")

            except Exception as exc:
                st.session_state.error = f"{exc}\n\n```\n{traceback.format_exc()}\n```"
                status.update(label="❌ Pipeline error", state="error")

if st.session_state.pipeline_log:
    with tracker_slot: render_pipeline(st.session_state.pipeline_log)


# ══════════════════════════════════════════════════════════════════════════════
# ERROR
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.error:
    st.error("Pipeline failed:")
    st.markdown(st.session_state.error)


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.result:
    res = st.session_state.result

    title     = _plan_attr(res,"blog_title") or res.get("topic","Article")
    blog_kind = (_plan_attr(res,"blog_kind","explainer") or "").replace("_"," ").title()
    audience  = _plan_attr(res,"audience","—")
    tone_val  = _plan_attr(res,"tone","—")
    mode      = (res.get("mode") or "closed_book")
    final_md  = res.get("final") or ""
    wc        = len(final_md.split())
    evidence  = _evidence_dicts(res)

    # ── Safe data extraction — handles nested or flat ──────────────────────
    # After _extract_state the keys ARE flat: res["seo_audit"] = {...dict...}
    # But just in case they came in wrapped, we defensively unwrap:
    def _safe_dict(val) -> dict:
        if isinstance(val, dict): return val
        if hasattr(val, "model_dump"): return val.model_dump()
        return {}

    seo = _safe_dict(res.get("seo_audit"))
    fc  = _safe_dict(res.get("fact_check"))

    seo_score  = seo.get("score")
    rel_raw    = fc.get("overall_reliability")
    rel_pct    = round((rel_raw or 0)*100)
    read_time  = seo.get("estimated_read_time_minutes") or max(1,round(wc/200))

    # gauge colour
    def _gc(v, thresholds=(70,45)):
        if v is None: return "gc-amber"
        return "gc-green" if v>=thresholds[0] else ("gc-amber" if v>=thresholds[1] else "gc-red")

    def _cc(v, thresholds=(70,45)):
        if v is None: return ""
        return "c-green" if v>=thresholds[0] else ("c-amber" if v>=thresholds[1] else "c-red")

    # ── Metric bar ──────────────────────────────────────────────────────────
    seo_disp  = str(seo_score) if seo_score is not None else "—"
    seo_color = _cc(seo_score)
    rel_color = _cc(rel_pct)

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card mc-words">
        <div class="m-label">Words</div>
        <div class="m-value">{wc:,}</div>
        <div class="m-sub">{read_time} min read</div>
      </div>
      <div class="metric-card mc-seo">
        <div class="m-label">SEO Score</div>
        <div class="m-value {seo_color}">{seo_disp}</div>
        <div class="m-sub">out of 100</div>
      </div>
      <div class="metric-card mc-rel">
        <div class="m-label">Reliability</div>
        <div class="m-value {rel_color}">{rel_pct}%</div>
        <div class="m-sub">fact-checked</div>
      </div>
      <div class="metric-card mc-mode">
        <div class="m-label">Mode</div>
        <div class="m-value" style="font-size:15px;padding-top:4px">{mode.replace('_',' ').title()}</div>
        <div class="m-sub">{len(evidence)} source{'s' if len(evidence)!=1 else ''}</div>
      </div>
      <div class="metric-card mc-type">
        <div class="m-label">Type</div>
        <div class="m-value" style="font-size:14px;padding-top:4px">{blog_kind}</div>
        <div class="m-sub">{(audience or '')[:22]}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ────────────────────────────────────────────────────────────────
    t_ms,t_ed,t_st,t_seo,t_fc,t_res,t_crit,t_raw = st.tabs([
        "📄 Manuscript","✏️ AI Editor","🎯 Strategy",
        "📈 SEO Audit","✅ Fact Check","🔍 Research","🎯 Critic Scores","⚙️ Raw State"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — Manuscript
    # ════════════════════════════════════════════════════════════════════════
    with t_ms:
        st.markdown(f"## {title}")
        c1,c2,c3 = st.columns(3)
        with c1:
            st.download_button("📥 Download .md", final_md,
                f"{_slug(title)}.md","text/markdown",use_container_width=True)
        with c2:
            st.download_button("🌐 Download .html", build_html(res),
                f"{_slug(title)}.html","text/html",use_container_width=True)
        with c3:
            st.download_button("📋 Meta description",
                seo.get("suggested_meta_description","No meta description."),
                "meta.txt","text/plain",use_container_width=True)
        st.divider()
        st.markdown(final_md or "*No content generated.*", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — AI Editor
    # ════════════════════════════════════════════════════════════════════════
    with t_ed:
        undo_stack = st.session_state.undo_stack

        if undo_stack:
            uc1,uc2 = st.columns([2,5])
            with uc1:
                if st.button("↩ Undo last edit", use_container_width=True):
                    st.session_state.result["final"] = undo_stack.pop()
                    if len(st.session_state.editor_history)>=2:
                        st.session_state.editor_history = st.session_state.editor_history[:-2]
                    st.rerun()
            with uc2:
                st.markdown(
                    f'<div class="undo-bar">'
                    f'<span>📚 {len(undo_stack)} snapshot{"s" if len(undo_stack)!=1 else ""} saved</span>'
                    f'<span>Undo to roll back the last edit</span></div>',
                    unsafe_allow_html=True)

        # Quick actions — real Streamlit buttons ONLY
        st.markdown('<div class="section-header">Quick actions</div>', unsafe_allow_html=True)
        qa_cols = st.columns(4)
        triggered = None
        for i,(label,prompt) in enumerate(QUICK_ACTIONS):
            with qa_cols[i%4]:
                if st.button(label, key=f"qa_{i}", use_container_width=True):
                    triggered = prompt

        st.markdown('<div class="section-header">Conversation</div>', unsafe_allow_html=True)

        hist = st.session_state.editor_history
        if not hist:
            st.markdown(
                '<div class="chat-empty">'
                '<div class="chat-empty-title">Talk to your article</div>'
                '<div class="chat-empty-hint">'
                '"Make the intro simpler" · "Which section is weakest?"<br>'
                '"Add a code example to section 3" · "Rewrite the conclusion"'
                '</div></div>', unsafe_allow_html=True)
        else:
            parts = ['<div class="chat-wrap">']
            for turn in hist:
                safe = (turn["content"]
                        .replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
                safe = re.sub(r"`([^`]+)`", r"<code>\1</code>", safe)
                if turn["role"]=="user":
                    parts.append(f'<div class="cmsg user">'
                                 f'<div class="cavatar av-user">you</div>'
                                 f'<div class="cbubble user">{safe}</div></div>')
                else:
                    badge = ('<div class="edit-badge">✅ Article updated</div>'
                             if turn.get("action")=="edit" else "")
                    parts.append(f'<div class="cmsg">'
                                 f'<div class="cavatar av-ai">AI</div>'
                                 f'<div class="cbubble ai">{badge}{safe}</div></div>')
            parts.append('</div>')
            st.markdown("\n".join(parts), unsafe_allow_html=True)

        user_input = st.chat_input(
            "Edit, ask, or suggest… e.g. 'Rewrite the conclusion with more impact'")
        if triggered: user_input = triggered

        if user_input:
            st.session_state.undo_stack.append(
                st.session_state.result.get("final",""))
            st.session_state.editor_history.append(
                {"role":"user","content":user_input,"action":None})
            with st.spinner("✍️ Working on it…"):
                try:
                    out = _run_editor(user_input,
                                      st.session_state.result.get("final",""),
                                      st.session_state.editor_history[:-1])
                    if out["new_article"]:
                        st.session_state.result["final"] = out["new_article"]
                        # Re-run SEO + fact check on the updated article
                        try:
                            updated_state = dict(st.session_state.result)
                            updated_state["final"] = out["new_article"]
                            seo_result = seo_audit_node(updated_state)
                            fc_result  = fact_checker_node(updated_state)
                            st.session_state.result.update(seo_result)
                            st.session_state.result.update(fc_result)
                        except Exception as _seo_exc:
                            log.warning("post-edit re-audit failed: %s", _seo_exc)
                        st.session_state.editor_history.append(
                            {"role":"assistant",
                             "content":out["reply"] or out["change_summary"],
                             "action":"edit"})
                    else:
                        st.session_state.undo_stack.pop()
                        st.session_state.editor_history.append(
                            {"role":"assistant","content":out["reply"],"action":"chat"})
                except Exception as exc:
                    st.session_state.undo_stack.pop()
                    st.session_state.editor_history.append(
                        {"role":"assistant","content":f"❌ Error: {exc}","action":"error"})
            st.rerun()

        if hist:
            st.markdown("---")
            with st.expander("📄 Preview current article", expanded=False):
                st.markdown(st.session_state.result.get("final",""), unsafe_allow_html=True)
            dc1,dc2 = st.columns(2)
            with dc1:
                st.download_button("📥 Download edited .md",
                    st.session_state.result.get("final",""),
                    f"{_slug(title)}_edited.md","text/markdown",use_container_width=True)
            with dc2:
                st.download_button("🌐 Download edited .html",
                    build_html(st.session_state.result),
                    f"{_slug(title)}_edited.html","text/html",use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — Strategy
    # ════════════════════════════════════════════════════════════════════════
    with t_st:
        c1,c2 = st.columns(2)
        c1.metric("Audience", audience)
        c2.metric("Tone", tone_val)
        df = _tasks_df(res)
        if df is not None:
            st.dataframe(df, use_container_width=True, hide_index=True,
                column_config={
                    "id":                st.column_config.NumberColumn("ID",    width=50),
                    "title":             st.column_config.TextColumn("Section", width=240),
                    "target_words":      st.column_config.NumberColumn("Words", width=80),
                    "requires_research": st.column_config.CheckboxColumn("Research?"),
                    "requires_code":     st.column_config.CheckboxColumn("Code?"),
                })
        constraints = _plan_attr(res,"constraints",[]) or []
        if constraints:
            st.markdown("**Writing constraints**")
            for c in constraints: st.markdown(f"- {c}")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — SEO Audit
    # ════════════════════════════════════════════════════════════════════════
    with t_seo:
        if not seo or "score" not in seo:
            st.warning("SEO audit data is not yet available in this result. "
                       "It is generated as the last step of the pipeline.")
        else:
            score = seo["score"]
            gcls  = _gc(score)
            cg,cd = st.columns([1,2])
            with cg:
                st.markdown(f'<div class="gauge-wrap">'
                            f'<div class="gauge-num {gcls}">{score}</div>'
                            f'<div class="gauge-lbl">SEO Score</div></div>',
                            unsafe_allow_html=True)
                st.markdown("")
                for lbl,val in [("✅ Title length OK" if seo.get("title_length_ok") else "❌ Title too long/short", True),
                                 ("✅ Keyword density OK" if seo.get("keyword_density_ok") else "❌ Keyword density issue", True),
                                 ("✅ Clear headings" if seo.get("has_clear_headings") else "❌ Headings need work", True)]:
                    st.markdown(lbl)
            with cd:
                if seo.get("summary"):
                    st.info(seo["summary"])
                st.caption(f"⏱ Estimated read time: **{seo.get('estimated_read_time_minutes','?')} min**")
                kws = seo.get("suggested_keywords",[]) or []
                if kws:
                    st.markdown("**Suggested keywords**")
                    st.markdown(" ".join(f'<span class="kw-badge">{k}</span>' for k in kws),
                                unsafe_allow_html=True)
                meta_desc = seo.get("suggested_meta_description","")
                if meta_desc:
                    st.markdown("**Meta description**")
                    st.code(meta_desc, language=None)

            issues = seo.get("issues",[]) or []
            if issues:
                st.markdown('<div class="section-header">Issues</div>', unsafe_allow_html=True)
                for iss in issues:
                    sev   = iss.get("severity","low")
                    chipc = {"high":"chip-high","medium":"chip-medium","low":"chip-low"}.get(sev,"chip-low")
                    st.markdown(
                        f'<span class="chip {chipc}">{sev.upper()}</span> '
                        f'**{iss.get("issue","")}** — {iss.get("suggestion","")}',
                        unsafe_allow_html=True)
            else:
                st.success("🎉 No SEO issues found.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 5 — Fact Check
    # ════════════════════════════════════════════════════════════════════════
    with t_fc:
        if not fc or "overall_reliability" not in fc:
            st.warning("Fact-check data is not yet available in this result.")
        else:
            rel  = fc["overall_reliability"]
            pct  = round(rel*100)
            gcls = _gc(pct, thresholds=(70,40))
            cr,cs = st.columns([1,2])
            with cr:
                st.markdown(f'<div class="gauge-wrap">'
                            f'<div class="gauge-num {gcls}">{pct}%</div>'
                            f'<div class="gauge-lbl">Reliability</div></div>',
                            unsafe_allow_html=True)
            with cs:
                if fc.get("summary"):
                    st.info(fc["summary"])

            verdicts = fc.get("verdicts",[]) or []
            if verdicts:
                st.markdown('<div class="section-header">Claim verdicts</div>',
                            unsafe_allow_html=True)
                for v in verdicts:
                    supported = v.get("supported",False)
                    conf      = v.get("confidence",0)
                    bw        = round(conf*100)
                    bc        = "#3FB950" if supported else "#F85149"
                    icon      = "✅" if supported else "⚠️"
                    with st.expander(f"{icon} {v.get('claim','')[:88]}"):
                        st.markdown(
                            f'<div class="conf-wrap">'
                            f'<span style="font-size:12px;color:#484F58;white-space:nowrap">Confidence</span>'
                            f'<div class="conf-bg"><div class="conf-fill" '
                            f'style="width:{bw}%;background:{bc}"></div></div>'
                            f'<span style="font-size:13px;font-weight:600;color:#E6EDF3">{bw}%</span>'
                            f'</div>', unsafe_allow_html=True)
                        if v.get("supporting_url"):
                            st.markdown(f"🔗 [Source]({v['supporting_url']})")
                        if v.get("note"):
                            st.caption(v["note"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 6 — Research
    # ════════════════════════════════════════════════════════════════════════
    with t_res:
        # ✅ FIX: check actual evidence list, NOT mode string
        mode_str = (mode or "").lower().replace(" ","_")

        if evidence:
            st.caption(f"🔍 {len(evidence)} source{'s' if len(evidence)!=1 else ''} gathered")
            for item in evidence:
                with st.expander(f"🔗 {item.get('title','Untitled')}"):
                    meta = " · ".join(p for p in [item.get("source"),item.get("published_at")] if p)
                    if meta: st.caption(meta)
                    if item.get("snippet"): st.write(item["snippet"])
                    url = item.get("url","")
                    if url and url.startswith("http"):
                        st.link_button("Visit source", url)
        else:
            if mode_str == "closed_book":
                st.info("ℹ️ **Closed book mode** — wrote from training knowledge, no web research needed.")
            else:
                st.info("ℹ️ No sources gathered. This can happen with very broad or ambiguous topics.")

        queries = res.get("queries") or []
        if queries:
            with st.expander(f"🔎 Search queries used ({len(queries)})"):
                for q in queries: st.markdown(f"- `{q}`")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 7 — Critic Scores
    # ════════════════════════════════════════════════════════════════════════
    with t_crit:
        critic_scores = res.get("critic_scores") or []
        style_ctx     = res.get("style_context") or ""

        if style_ctx:
            with st.expander("🧠 Writer memory — style context used", expanded=False):
                st.caption("These past articles were retrieved and used to match your writing style.")
                st.markdown(style_ctx)
        else:
            st.info("🧠 No past articles in memory yet — after this run, your style will be saved for future articles.")

        if not critic_scores:
            st.info("No critic scores available — run the pipeline to generate them.")
        else:
            st.markdown('<div class="section-header">Section quality scores</div>', unsafe_allow_html=True)

            passed  = sum(1 for s in critic_scores if s.get("passed", True))
            revised = sum(1 for s in critic_scores if not s.get("passed", True))

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Sections written", len(critic_scores))
            mc2.metric("Passed first time", passed)
            mc3.metric("Revised by critic", revised, delta=f"-{revised} revisions needed" if revised else "✅ All passed")

            st.markdown("---")
            for score in sorted(critic_scores, key=lambda x: x.get("overall", 0)):
                overall  = score.get("overall", 0)
                passed_s = score.get("passed", True)
                icon     = "✅" if passed_s else "🔄"
                title    = score.get("section_title", f"Section {score.get('section_id','?')}")
                bar_w    = round(overall * 10)
                bar_col  = "#3FB950" if overall >= 7 else ("#D29922" if overall >= 5 else "#F85149")

                with st.expander(f"{icon} {title} — overall {overall:.1f}/10"):
                    # Radar-style dimension bars
                    dims = [("Accuracy", score.get("accuracy",0)),
                            ("Depth",    score.get("depth",0)),
                            ("Clarity",  score.get("clarity",0)),
                            ("Grounding",score.get("grounding",0))]
                    for dim_name, dim_val in dims:
                        dw = round(dim_val * 10)
                        dc = "#3FB950" if dim_val>=7 else ("#D29922" if dim_val>=5 else "#F85149")
                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:10px;margin:4px 0">'
                            f'<span style="width:80px;font-size:12px;color:#8B949E">{dim_name}</span>'
                            f'<div style="flex:1;height:6px;background:#21262D;border-radius:3px">'
                            f'<div style="width:{dw}%;height:6px;border-radius:3px;background:{dc}"></div></div>'
                            f'<span style="font-size:12px;font-weight:500;color:#E6EDF3">{dim_val:.1f}</span>'
                            f'</div>', unsafe_allow_html=True)

                    if not passed_s and score.get("feedback"):
                        st.markdown("**Critic feedback that triggered revision:**")
                        st.warning(score["feedback"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 8 — Raw State
    # ════════════════════════════════════════════════════════════════════════
    with t_raw:
        st.subheader("Full pipeline state")
        try:    st.json(_serialisable(res))
        except: st.text(str(res))


# ══════════════════════════════════════════════════════════════════════════════
# WELCOME
# ══════════════════════════════════════════════════════════════════════════════

else:
    if not st.session_state.pipeline_log:
        st.markdown("""
        <div class="welcome-wrap">
          <div class="welcome-title">DeepWrite v2</div>
          <div class="welcome-sub">
            A multi-agent AI pipeline that researches, plans, writes, fact-checks,
            and SEO-audits a full technical blog post — completely end to end.
          </div>
          <div class="fgrid">
            <div class="fitem">
              <div class="ft" style="color:#58A6FF">🔀 Intelligent Router</div>
              <div class="fd">Classifies your topic and decides if live web research is needed before writing.</div>
            </div>
            <div class="fitem">
              <div class="ft" style="color:#3FB950">✍️ Parallel Workers</div>
              <div class="fd">Multiple LLM agents write each section simultaneously, then merged into one draft.</div>
            </div>
            <div class="fitem">
              <div class="ft" style="color:#D29922">✅ Fact Checker</div>
              <div class="fd">Every factual claim cross-referenced against gathered evidence with a confidence score.</div>
            </div>
            <div class="fitem">
              <div class="ft" style="color:#BC8CFF">📈 SEO Audit</div>
              <div class="fd">Automatic scoring, keyword suggestions, meta description, and prioritised issue list.</div>
            </div>
            <div class="fitem">
              <div class="ft" style="color:#FF7B72">✏️ AI Editor</div>
              <div class="fd">Chat with your article — natural language edits, questions, and quick-action buttons.</div>
            </div>
            <div class="fitem">
              <div class="ft" style="color:#79C0FF">📂 Draft History</div>
              <div class="fd">Every run auto-saved to SQLite. Reload any past article from the sidebar.</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
