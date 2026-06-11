"""
miniRAG — Streamlit UI
Production-grade RAG with chat history, streaming, hybrid retrieval, and latency breakdown.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ── Path setup: resolve src/minirag package ──────────────────────────────────
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src" / "minirag"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
# Also keep repo-root flat imports working (legacy)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env", override=True)

import streamlit as st

# ── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="miniRAG · Gemini + Pinecone",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/sakshinaithani2005/miniRAG",
        "Report a bug": "https://github.com/sakshinaithani2005/miniRAG/issues",
        "About": "miniRAG — production RAG with Gemini, Pinecone, hybrid retrieval & RAGAS eval.",
    },
)

# ── Custom CSS — dark glassmorphism theme ────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* === Background === */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #131a3a 50%, #1a0533 100%);
    min-height: 100vh;
}

/* === Glassmorphism cards === */
.glass-card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s ease;
}
.glass-card:hover { border-color: rgba(139,92,246,0.5); }

/* === Sidebar === */
[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.85) !important;
    border-right: 1px solid rgba(139,92,246,0.25) !important;
    backdrop-filter: blur(20px);
}

/* === Primary button === */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #8b5cf6, #6366f1) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(139,92,246,0.45) !important;
}

/* === Chat messages === */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    margin-bottom: 0.75rem !important;
    animation: fadeSlideIn 0.3s ease !important;
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* === Chat input === */
[data-testid="stChatInput"] > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(139,92,246,0.4) !important;
    border-radius: 12px !important;
}

/* === Metrics === */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    border: 1px solid rgba(255,255,255,0.08);
}

/* === Expander === */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
}

/* === Badges / tags === */
.tag {
    display: inline-block;
    background: rgba(139,92,246,0.2);
    border: 1px solid rgba(139,92,246,0.5);
    color: #c4b5fd;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.4px;
    margin-right: 6px;
}
.tag-web {
    background: rgba(16,185,129,0.2);
    border-color: rgba(16,185,129,0.5);
    color: #6ee7b7;
}

/* === Latency bar === */
.latency-bar-wrap { margin: 0.5rem 0 1rem; }
.latency-label { font-size: 0.72rem; color: #a78bfa; margin-bottom: 4px; }
.latency-bar {
    height: 6px;
    border-radius: 4px;
    background: linear-gradient(90deg, #8b5cf6, #6366f1);
    margin-bottom: 6px;
    transition: width 0.5s ease;
}

/* === Warning === */
.grounding-warn {
    background: rgba(234,179,8,0.1);
    border: 1px solid rgba(234,179,8,0.4);
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    color: #fde047;
    font-size: 0.82rem;
    margin-top: 0.5rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Secret resolution ────────────────────────────────────────────────────────
def _secret(key: str, default: str | None = None) -> str | None:
    try:
        v = st.secrets.get(key)
        if v:
            return v
    except Exception:
        pass
    return os.getenv(key, default)


GOOGLE_API_KEY = _secret("GOOGLE_API_KEY")
PINECONE_API_KEY = _secret("PINECONE_API_KEY")
PINECONE_INDEX_NAME = _secret("PINECONE_INDEX_NAME", "mini-rag")

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    st.error(
        "❌ **Missing API keys.**\n\n"
        "Add `GOOGLE_API_KEY` and `PINECONE_API_KEY` to your `.env` file or Streamlit Secrets."
    )
    st.stop()

# Propagate keys so downstream modules find them via os.getenv
os.environ.setdefault("GOOGLE_API_KEY", GOOGLE_API_KEY)
os.environ.setdefault("PINECONE_API_KEY", PINECONE_API_KEY)
os.environ.setdefault("PINECONE_INDEX_NAME", PINECONE_INDEX_NAME)

# ── Lazy imports (after env is ready) ────────────────────────────────────────
from config import Config, RetrievalStrategy
from document_processor import process_documents
from embeddings import get_embeddings
from llm import get_llm
from observability import QueryTracer, configure_logging
from rag_chain import create_rag_chain, stream_rag
from retriever import create_retriever
from vectorstore import add_documents_to_vectorstore, get_vectorstore

configure_logging()

# ── Session state bootstrap ───────────────────────────────────────────────────
_DEFAULTS = {
    "chat_history": [],          # list[dict] — {role, content, sources, latency, warnings}
    "indexed": False,
    "chunks": [],
    "chunks_count": 0,
    "doc_name": "",
    "feedback": {},              # {msg_idx: "up"|"down"}
    "strategy": RetrievalStrategy.HYBRID.value,
}
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ── Cached component initialisation ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _init_components():
    """Load heavy singletons once per server process."""
    return {
        "embeddings": get_embeddings(),
        "llm": get_llm(),
        "vectorstore": get_vectorstore(),
    }


try:
    components = _init_components()
except Exception as exc:
    st.error(f"❌ **Initialisation failed:** {exc}")
    st.stop()


# ── Helper: render latency bars ───────────────────────────────────────────────
def _latency_bars(latency: dict) -> str:
    total = latency.get("total_ms", 1) or 1
    rows = ""
    for stage, label in [
        ("retrieve_ms", "🔍 Retrieve"),
        ("rerank_ms",   "📐 Rerank"),
        ("generate_ms", "✨ Generate"),
    ]:
        ms = latency.get(stage, 0)
        pct = min(int(ms / total * 100), 100)
        rows += (
            f'<div class="latency-label">{label} — {ms:.0f} ms</div>'
            f'<div class="latency-bar" style="width:{pct}%"></div>'
        )
    rows += f'<div style="font-size:0.75rem;color:#6b7280;">Total: {total:.0f} ms</div>'
    return f'<div class="latency-bar-wrap">{rows}</div>'


# ── Header ───────────────────────────────────────────────────────────────────
col_title, col_badges = st.columns([3, 1])
with col_title:
    st.markdown(
        "<h1 style='margin-bottom:0;color:#e2d9f3;'>🧠 miniRAG</h1>"
        "<p style='color:#a78bfa;margin-top:4px;'>Gemini · Pinecone · Hybrid Retrieval · RAGAS Eval</p>",
        unsafe_allow_html=True,
    )
with col_badges:
    st.markdown(
        '<div style="text-align:right;padding-top:1rem;">'
        '<span class="tag">gemini-2.5-flash</span>'
        '<span class="tag">FlashRank</span>'
        '</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Document ingestion + settings
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📄 Load Document")
    st.caption("Upload a file or paste text to index it into Pinecone.")

    uploaded_file = st.file_uploader(
        "PDF / TXT / DOCX",
        type=["pdf", "txt", "docx"],
        label_visibility="collapsed",
    )
    input_text = st.text_area("…or paste text", height=120, placeholder="Paste your document here")

    st.markdown("---")
    st.markdown("### ⚙️ Retrieval Settings")
    chosen_strategy = st.radio(
        "Strategy",
        options=[s.value for s in RetrievalStrategy],
        index=1,  # default: hybrid
        format_func=lambda s: {"dense": "Dense (cosine)", "hybrid": "Hybrid (BM25 + RRF)", "mmr": "MMR (diversity)"}[s],
        horizontal=False,
    )
    st.session_state.strategy = chosen_strategy

    enable_rewrite = st.toggle("Query rewriting (HyDE-lite)", value=True)
    enable_web_fallback = st.toggle("Web search fallback", value=False)

    st.markdown("---")

    if st.button("📤 Index Document", use_container_width=True, type="primary"):
        if not input_text and not uploaded_file:
            st.warning("Provide text or upload a file first.")
        else:
            with st.spinner("Chunking, embedding & indexing…"):
                try:
                    t0 = time.perf_counter()
                    chunks = process_documents(
                        uploaded_file=uploaded_file,
                        input_text=input_text or None,
                    )
                    num = add_documents_to_vectorstore(chunks, components["vectorstore"])
                    elapsed = time.perf_counter() - t0

                    st.session_state.chunks = chunks
                    st.session_state.chunks_count = num
                    st.session_state.indexed = True
                    st.session_state.doc_name = (
                        uploaded_file.name if uploaded_file else "Pasted text"
                    )
                    st.session_state.chat_history = []  # fresh chat for new doc

                    st.success(f"✅ {num} chunks indexed in {elapsed:.2f}s")
                except Exception as exc:
                    st.error(f"❌ **Indexing failed:** {exc}")

    if st.session_state.indexed:
        st.info(
            f"📚 **{st.session_state.doc_name}**\n\n"
            f"{st.session_state.chunks_count} chunks · "
            f"{chosen_strategy} retrieval"
        )

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ────────────────────────────────────────────────────────────────────────────
# MAIN — Chat interface
# ────────────────────────────────────────────────────────────────────────────
if not st.session_state.indexed:
    st.markdown(
        '<div class="glass-card">'
        "<h3 style='color:#e2d9f3;'>👋 Welcome to miniRAG</h3>"
        "<p style='color:#9ca3af;'>Upload a PDF, TXT, or DOCX in the sidebar — "
        "or paste raw text — then ask questions and get grounded, cited answers.</p>"
        "<ul style='color:#6b7280;'>"
        "<li>🔍 <b>Hybrid BM25 + dense retrieval</b> for best coverage</li>"
        "<li>📐 <b>FlashRank reranking</b> for precision</li>"
        "<li>✍️ <b>Query rewriting</b> for ambiguous questions</li>"
        "<li>📊 <b>RAGAS evaluation</b> — run <code>python eval/eval_pipeline.py</code></li>"
        "</ul>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ── Render existing chat history ─────────────────────────────────────────────
for idx, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"], avatar="🧠" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            # Latency breakdown
            if msg.get("latency"):
                with st.expander("⏱ Latency breakdown", expanded=False):
                    st.markdown(_latency_bars(msg["latency"]), unsafe_allow_html=True)

            # Sources
            if msg.get("sources"):
                with st.expander(f"📚 Sources ({len(msg['sources'])} chunks)", expanded=False):
                    for i, doc in enumerate(msg["sources"], 1):
                        source_type = doc.metadata.get("source_type", "")
                        tag_cls = "tag-web" if source_type == "web_search" else "tag"
                        label = "🌐 Web" if source_type == "web_search" else f"[{i}]"
                        st.markdown(
                            f'<span class="{tag_cls}">{label}</span> '
                            f"**{doc.metadata.get('source', 'Unknown')}**",
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            doc.metadata.get("source_snippet", doc.page_content[:200])
                        )
                        if "page" in doc.metadata:
                            st.caption(f"Page {doc.metadata['page']}")

            # Grounding warnings
            for w in msg.get("warnings", []):
                st.markdown(f'<div class="grounding-warn">{w}</div>', unsafe_allow_html=True)

            # Feedback
            fb_key = f"fb_{idx}"
            current_fb = st.session_state.feedback.get(idx)
            c1, c2, _ = st.columns([1, 1, 8])
            with c1:
                if st.button("👍", key=f"up_{idx}", disabled=current_fb is not None):
                    st.session_state.feedback[idx] = "up"
                    st.rerun()
            with c2:
                if st.button("👎", key=f"dn_{idx}", disabled=current_fb is not None):
                    st.session_state.feedback[idx] = "down"
                    st.rerun()
            if current_fb:
                st.caption("✅ Thanks for your feedback!" if current_fb == "up" else "📝 Noted — we'll improve.")

# ── Chat input ───────────────────────────────────────────────────────────────
query = st.chat_input("Ask a question about the document…")

if query:
    # Store user message
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(query)

    # Build retriever for this turn
    strategy_enum = RetrievalStrategy(st.session_state.strategy)
    retriever = create_retriever(
        components["vectorstore"],
        corpus=st.session_state.chunks or None,
        strategy=strategy_enum,
    )
    chain = create_rag_chain(retriever, components["llm"])
    tracer = QueryTracer(query)

    with st.chat_message("assistant", avatar="🧠"):
        with tracer.stage("generate"):
            token_stream, retrieved_docs = stream_rag(
                chain,
                retriever,
                query,
                enable_rewrite=enable_rewrite,
                llm=components["llm"] if enable_rewrite else None,
            )

            # Web fallback
            if enable_web_fallback:
                from web_search import augment_with_web
                retrieved_docs = augment_with_web(retrieved_docs, query)

            answer = st.write_stream(token_stream)

        latency_dict = tracer.latency.as_dict()

        # Grounding check
        from rag_chain import verify_citations
        warnings = verify_citations(answer, len(retrieved_docs))

        # Latency expander
        with st.expander("⏱ Latency breakdown", expanded=False):
            st.markdown(_latency_bars(latency_dict), unsafe_allow_html=True)

        # Sources expander
        if retrieved_docs:
            with st.expander(f"📚 Sources ({len(retrieved_docs)} chunks)", expanded=False):
                for i, doc in enumerate(retrieved_docs, 1):
                    source_type = doc.metadata.get("source_type", "")
                    tag_cls = "tag-web" if source_type == "web_search" else "tag"
                    label = "🌐 Web" if source_type == "web_search" else f"[{i}]"
                    st.markdown(
                        f'<span class="{tag_cls}">{label}</span> '
                        f"**{doc.metadata.get('source', 'Unknown')}**",
                        unsafe_allow_html=True,
                    )
                    st.caption(doc.metadata.get("source_snippet", doc.page_content[:200]))

        # Grounding warnings
        for w in warnings:
            st.markdown(f'<div class="grounding-warn">{w}</div>', unsafe_allow_html=True)

        # Feedback buttons
        msg_idx = len(st.session_state.chat_history)
        c1, c2, _ = st.columns([1, 1, 8])
        with c1:
            st.button("👍", key=f"up_{msg_idx}")
        with c2:
            st.button("👎", key=f"dn_{msg_idx}")

    # Persist assistant message
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": retrieved_docs,
        "latency": latency_dict,
        "warnings": warnings,
    })

    tracer.log(num_docs=len(retrieved_docs), strategy=st.session_state.strategy)
