"""
app.py — Curator: a NotebookLM-style web UI (Streamlit)
-------------------------------------------------------
A three-panel notebook over your PDFs:

  ┌── Sources ──┬────── Chat ──────┬── Studio ──┐
  │ add / pick  │ streaming answers │ summary    │
  │ PDFs        │ with [1] citations│ study guide│
  │             │ + copy/redo/chart │ FAQ …      │
  └─────────────┴───────────────────┴────────────┘

Everything runs locally (Ollama + sentence-transformers + ChromaDB). No API key.
Run it with:  streamlit run app.py
"""

import json
import re

import pandas as pd
import plotly.express as px
import streamlit as st

from rag_engine import NOT_FOUND, RagEngine, RagError, warm_up


st.set_page_config(page_title="Curator — your sources, curated",
                   page_icon="🏛️", layout="wide")

# --- Museum palette (charcoal + gold) -------------------------------------
GOLD = "#C9A24B"
GOLD_HI = "#E3C46A"
INK = "#ECE6DA"
CHARCOAL = "#13110E"
PANEL = "#1C1915"
BORDER = "#332E25"

# Plotly styling — gold-forward, warm museum palette.
px.defaults.template = "plotly_dark"
PALETTE = ["#C9A24B", "#7FA7A0", "#C77B54", "#9C8AB0", "#6E8CBF", "#D9C27A"]
px.defaults.color_discrete_sequence = PALETTE

# Curator theme: charcoal + gold, serif display headings, refined chat + citations.
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', system-ui, sans-serif; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 1rem; max-width: 1500px; }}

/* Branded header */
.curator-head {{ display:flex; align-items:baseline; gap:.6rem; margin-bottom:.1rem; }}
.curator-logo {{ font-family:'Fraunces', Georgia, serif; font-weight:600;
  font-size:2.1rem; color:{INK}; letter-spacing:.01em; }}
.curator-logo .dot {{ color:{GOLD}; }}
.curator-sub {{ color:#9c9384; font-size:.92rem; font-style:italic; }}
.curator-rule {{ height:2px; margin:.55rem 0 1.1rem;
  background:linear-gradient(90deg,{GOLD} 0%, rgba(201,162,75,.15) 45%, transparent 80%); }}

/* Panel headers */
.panel-title {{ font-family:'Fraunces', Georgia, serif; font-size:1rem; font-weight:600;
  letter-spacing:.02em; color:{GOLD}; margin:.1rem 0 .7rem;
  border-bottom:1px solid {BORDER}; padding-bottom:.4rem; }}

/* Chat bubbles */
[data-testid="stChatMessage"] {{ border-radius:14px; border:1px solid {BORDER};
  background:{PANEL}; padding:.35rem .7rem; }}

/* Bordered containers (overview / studio) */
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius:14px; }}

/* Buttons */
.stButton > button {{ border-radius:10px; border:1px solid {BORDER};
  transition:all .15s ease; }}
.stButton > button:hover {{ border-color:{GOLD}; color:{GOLD_HI}; }}

/* Citation badges with hover popover */
sup.cite {{ color:{CHARCOAL}; font-weight:700; background:{GOLD};
  padding:1px 6px; border-radius:6px; margin:0 2px; font-size:.72em;
  cursor:help; position:relative; top:-.15em; }}
sup.cite .cite-pop {{ display:none; position:absolute; left:0; bottom:160%;
  z-index:1000; width:300px; padding:10px 12px; background:#0d0b09;
  border:1px solid {GOLD}; border-radius:10px; color:{INK}; font-weight:400;
  font-size:.85rem; line-height:1.4; white-space:normal;
  box-shadow:0 10px 30px rgba(0,0,0,.6); }}
sup.cite:hover .cite-pop {{ display:block; }}

/* Confidence chip + captions */
.stCaption, [data-testid="stCaptionContainer"] {{ color:#9c9384; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Warming up the local AI models…")
def _boot():
    warm_up()
    return RagEngine()


engine = _boot()

# --- session state --------------------------------------------------------
ss = st.session_state
ss.setdefault("history", [])        # [{q, a, sources, viz, followups}]
ss.setdefault("overview", {})       # doc -> {summary, topics, questions}
ss.setdefault("notes", [])          # ["saved answer text", ...]
ss.setdefault("studio", {})         # kind -> markdown text
ss.setdefault("doc_filter", [])
ss.setdefault("pending", None)      # (question, temperature) queued for a run
ss.setdefault("studio_run", None)   # a studio kind to generate this run
ss.setdefault("copy_idx", None)     # which message's copy box is open


# ===========================================================================
# Helpers
# ===========================================================================
CONF = {"High": "🟢 High", "Medium": "🟡 Medium", "Low": "🔴 Low"}
_CITE_RE = re.compile(r"\[(\d{1,2})\]")


def _esc(t: str) -> str:
    return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def citationize(text: str, sources=None) -> str:
    """Turn [1] markers into hover-popover citation badges. On hover the badge
    shows the matching source passage (NotebookLM-style), no JS needed."""
    src = sources or []

    def repl(m):
        n = int(m.group(1))
        if 1 <= n <= len(src):
            s = src[n - 1]
            page = f" · p.{s['page']}" if s.get("page") else ""
            tip = _esc(f"[{n}] {s.get('source', '?')}{page} — {s['text'][:320]}")
            return (f"<sup class='cite'><span class='cite-pop'>{tip}</span>"
                    f"{n}</sup>")
        return f"<sup class='cite'>{n}</sup>"

    return _CITE_RE.sub(repl, text)


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"🔎 {len(sources)} sources"):
        for i, s in enumerate(sources, start=1):
            page = f" · p.{s['page']}" if s.get("page") else ""
            st.markdown(f"**[{i}] {s.get('source', '?')}**{page}")
            st.caption(s["text"])


def render_chart(viz):
    if not viz:
        return
    fmt, data = viz.get("format", "text"), viz.get("data")
    title = viz.get("title") or None
    try:
        if fmt in {"bar", "line", "pie"} and data:
            df = pd.DataFrame(data)
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value"])
            if df.empty:
                return
            if fmt == "bar":
                fig = px.bar(df, x="label", y="value", title=title)
            elif fmt == "line":
                fig = px.line(df, x="label", y="value", markers=True, title=title)
            else:
                fig = px.pie(df, names="label", values="value", title=title, hole=0.55)
            fig.update_layout(showlegend=(fmt == "pie"),
                              xaxis_title=viz.get("x_label") or None,
                              yaxis_title=viz.get("y_label") or None,
                              margin=dict(t=48, l=0, r=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        elif fmt == "table" and isinstance(data, dict) and data.get("rows"):
            st.dataframe(pd.DataFrame(data["rows"], columns=data.get("columns")),
                         use_container_width=True, hide_index=True)
    except Exception:
        pass


def render_turn(turn, idx, is_last):
    """Render a stored assistant turn: answer + citations + chart + actions."""
    st.markdown(citationize(turn["a"], turn.get("sources")), unsafe_allow_html=True)
    render_chart(turn.get("viz"))
    if turn.get("sources"):
        st.caption(f"Grounding confidence: {CONF.get(turn.get('confidence'), '⚪')}")
    render_sources(turn.get("sources"))

    # --- message action row ---
    c = st.columns(4)
    if c[0].button("📋 Copy", key=f"cp{idx}"):
        ss.copy_idx = None if ss.copy_idx == idx else idx
    if c[1].button("📌 Save note", key=f"nt{idx}"):
        ss.notes.append(turn["a"])
        st.toast("Saved to notes")
    if turn.get("sources") and c[2].button("📊 Visualize", key=f"vz{idx}"):
        with st.spinner("Building a chart…"):
            try:
                turn["viz"] = engine.visualize(turn["q"], turn["sources"])
            except RagError as e:
                st.error(str(e))
        st.rerun()
    if is_last and c[3].button("🔄 Regenerate", key=f"rg{idx}"):
        ss.history.pop()
        ss.pending = (turn["q"], 0.6)
        st.rerun()

    if ss.copy_idx == idx:
        st.code(turn["a"], language="markdown")   # has a native copy button

    # --- follow-up chips (only under the latest turn; computed on demand) ---
    if is_last and turn.get("sources"):
        fus = turn.get("followups")
        if fus is None:   # not computed yet -> offer a button
            if st.button("💡 Suggest follow-ups", key=f"sf{idx}"):
                with st.spinner("Thinking of follow-ups…"):
                    turn["followups"] = engine.suggest_followups(turn["q"], turn["a"])
                st.rerun()
        else:
            for j, fu in enumerate(fus):
                if st.button(f"💡 {fu}", key=f"fu{idx}_{j}"):
                    ss.pending = (fu, 0.0)
                    st.rerun()


def run_answer(question: str, temperature: float) -> bool:
    """Stream one answer into the chat and store the turn. Returns True on
    success (so the caller only reruns when a turn was actually appended)."""
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        status = st.status("Thinking…", expanded=False)
        sink = {}   # session-safe: answer_stream writes sources/clarify here
        try:
            text = st.write_stream(engine.answer_stream(
                question, history=ss.history, doc_filter=ss.doc_filter,
                temperature=temperature, sink=sink,
                step=lambda m: status.update(label=m)))
            status.update(label="Done", state="complete")
        except RagError as e:
            status.update(label="Failed", state="error")
            st.error(str(e))
            return False
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(f"Something went wrong: {e}")
            return False

        sources = list(sink.get("sources") or [])
        clarified = sink.get("clarify") is not None
        # followups start as None (= "not computed"); a clarify turn gets [].
        followups = [] if (clarified or text.strip() == NOT_FOUND or not sources) \
            else None
        ss.history.append({"q": question, "a": text, "sources": sources,
                           "viz": None, "followups": followups,
                           "confidence": _conf(sources)})
    return True


def _conf(sources):
    from rag_engine import _confidence
    return _confidence(sources) if sources else None


# ===========================================================================
# PANEL 1 — Sources
# ===========================================================================
def sources_panel():
    st.markdown("<div class='panel-title'>📚 Sources</div>", unsafe_allow_html=True)
    uploads = st.file_uploader("Add PDF(s)", type="pdf",
                               accept_multiple_files=True, label_visibility="collapsed")
    for up in uploads or []:
        b = up.getvalue()
        doc = engine.doc_id(b)
        if engine.has_document(doc):
            continue
        try:
            prog = st.progress(0.0, text=f"Indexing {up.name}…")
            res = engine.ingest_pdf(b, up.name, progress_callback=lambda d, t:
                                    prog.progress(d / t if t else 1.0,
                                                  text=f"Indexing {up.name}… {d}/{t}"))
            prog.empty()
            if res.get("truncated"):
                st.warning(f"'{up.name}': indexed first {res['pages_read']}/"
                           f"{res['total_pages']} pages.")
        except RagError as e:
            st.error(f"{up.name}: {e}")
        except Exception as e:
            st.error(f"{up.name}: {e}")

    docs = engine.list_documents()
    if not docs:
        st.info("Add a PDF to begin.")
        ss.doc_filter = []
        return

    st.caption("Search within:")
    selected = []
    for d in docs:
        row = st.columns([0.82, 0.18])
        # `value=True` only applies on a checkbox's FIRST render, so newly-added
        # docs default to checked while existing ones keep their toggled state.
        checked = row[0].checkbox(f"{d['source']}  ·  {d['chunks']} chunks",
                                  value=True, key=f"chk_{d['doc']}")
        if checked:
            selected.append(d["doc"])
        if row[1].button("🗑", key=f"del_{d['doc']}", help="Remove this source"):
            engine.delete_document(d["doc"])
            ss.overview.pop(d["doc"], None)
            st.rerun()
    ss.doc_filter = selected


# ===========================================================================
# PANEL 3 — Studio (overview + generated artifacts + notes)
# ===========================================================================
STUDIO_KINDS = [("summary", "📝 Summary"), ("study_guide", "🎓 Study guide"),
                ("faq", "❓ FAQ"), ("timeline", "🗓️ Timeline"),
                ("briefing", "🧾 Briefing")]


def studio_panel():
    st.markdown("<div class='panel-title'>🎛️ Studio</div>", unsafe_allow_html=True)

    if not engine.list_documents():
        st.caption("Studio unlocks once you add a source.")
        return

    # --- Overview for the (first) selected source (generated on demand) ---
    focus = ss.doc_filter[0] if ss.doc_filter else None
    if focus:
        with st.container(border=True):
            st.markdown("**Overview**")
            if focus not in ss.overview:
                if st.button("📖 Generate overview", key="gen_ov",
                             use_container_width=True):
                    with st.spinner("Reading the document…"):
                        try:
                            ss.overview[focus] = engine.document_overview(focus)
                        except Exception:
                            ss.overview[focus] = {"summary": "(unavailable)",
                                                  "topics": [], "questions": []}
                    st.rerun()
            else:
                ov = ss.overview[focus]
                st.write(ov["summary"])
                if ov["topics"]:
                    st.caption("Key topics: " + " · ".join(ov["topics"]))
                for i, q in enumerate(ov["questions"]):
                    if st.button(f"💬 {q}", key=f"ovq_{i}"):
                        ss.pending = (q, 0.0)
                        st.rerun()
        if len(ss.doc_filter) > 1:
            st.caption(f"(Overview shown for the first of {len(ss.doc_filter)} "
                       "selected sources.)")

    # --- Studio generators ---
    st.markdown("**Generate**")
    g = st.columns(2)
    for i, (kind, label) in enumerate(STUDIO_KINDS):
        if g[i % 2].button(label, key=f"gen_{kind}", use_container_width=True):
            ss.studio_run = kind
            st.rerun()

    # Run a requested generation (streamed).
    just_generated = None
    if ss.studio_run:
        just_generated = ss.studio_run
        ss.studio_run = None
        label = dict(STUDIO_KINDS)[just_generated]
        with st.container(border=True):
            st.markdown(f"**{label}**")
            try:
                text = st.write_stream(engine.studio_stream(just_generated, ss.doc_filter))
                ss.studio[just_generated] = text
            except RagError as e:
                st.error(str(e))

    # Show previously generated artifacts (but not the one just streamed above).
    for kind, label in STUDIO_KINDS:
        if kind in ss.studio and kind != just_generated:
            with st.expander(label):
                st.markdown(ss.studio[kind])
                st.download_button("⬇️", ss.studio[kind], file_name=f"{kind}.md",
                                   key=f"dl_{kind}")

    # --- Saved notes ---
    if ss.notes:
        st.markdown("**📌 Notes**")
        for i, note in enumerate(ss.notes):
            with st.expander(note[:48] + ("…" if len(note) > 48 else "")):
                st.markdown(note)
                if st.button("Delete", key=f"dn_{i}"):
                    ss.notes.pop(i)
                    st.rerun()
        st.download_button("⬇️ Export notes", "\n\n---\n\n".join(ss.notes),
                           file_name="notes.md")


# ===========================================================================
# PANEL 2 — Chat
# ===========================================================================
def chat_panel():
    st.markdown("<div class='panel-title'>💬 Chat</div>", unsafe_allow_html=True)

    if not engine.list_documents():
        st.info("⬅️ Add a PDF in the Sources panel to start chatting.")
        return

    # Replay history
    for idx, turn in enumerate(ss.history):
        with st.chat_message("user"):
            st.write(turn["q"])
        with st.chat_message("assistant"):
            render_turn(turn, idx, is_last=(idx == len(ss.history) - 1))

    if not ss.doc_filter:
        st.info("Select at least one source in the Sources panel to ask about it.")
        return

    # A queued question (from a chip / suggested question / regenerate)
    if ss.pending:
        q, temp = ss.pending
        ss.pending = None
        if run_answer(q, temp):
            st.rerun()

    typed = st.chat_input("Ask anything about your sources…")
    if typed:
        if run_answer(typed, 0.0):
            st.rerun()


# ===========================================================================
# Layout
# ===========================================================================
st.markdown(
    "<div class='curator-head'>"
    "<span class='curator-logo'>🏛️ Curator<span class='dot'>.</span></span>"
    "<span class='curator-sub'>your sources, curated &amp; explained</span>"
    "</div><div class='curator-rule'></div>",
    unsafe_allow_html=True,
)
left, center, right = st.columns([1.15, 2.5, 1.5], gap="medium")
with left:
    sources_panel()
# Render chat before Studio so Studio's (optional) overview call can't delay the
# primary chat flow.
with center:
    chat_panel()
with right:
    studio_panel()
