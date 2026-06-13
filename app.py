# ruff: noqa: E402
"""
miniRAG — Streamlit UI
Production-grade RAG with chat history, streaming, hybrid retrieval, and latency breakdown.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src" / "minirag"
# src/minirag MUST be first so it wins over stale flat-root files
for _p in [str(_SRC), str(_ROOT)]:
    if _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SRC))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env", override=True)

import streamlit as st

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
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

# ── Custom CSS — dark glassmorphism theme ─────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #131a3a 50%, #1a0533 100%);
    min-height: 100vh;
}
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

[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.85) !important;
    border-right: 1px solid rgba(139,92,246,0.25) !important;
    backdrop-filter: blur(20px);
}
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
[data-testid="stChatInput"] > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(139,92,246,0.4) !important;
    border-radius: 12px !important;
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    border: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
}
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
.latency-bar-wrap { margin: 0.5rem 0 1rem; }
.latency-label { font-size: 0.72rem; color: #a78bfa; margin-bottom: 4px; }
.latency-bar {
    height: 6px;
    border-radius: 4px;
    background: linear-gradient(90deg, #8b5cf6, #6366f1);
    margin-bottom: 6px;
    transition: width 0.5s ease;
}
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


# ── Secret resolution ─────────────────────────────────────────────────────────
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

# Use os.environ[] (not setdefault) so keys are always current
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["PINECONE_INDEX_NAME"] = PINECONE_INDEX_NAME

# ── Lazy imports (after env vars are set) ────────────────────────────────────
from config import RetrievalStrategy
from document_processor import process_documents
from llm import get_llm
from observability import QueryTracer, configure_logging
from rag_chain import create_rag_chain, verify_citations
from retriever import create_retriever
from vectorstore import add_documents_to_vectorstore, get_vectorstore

configure_logging()


# ── Session state bootstrap ───────────────────────────────────────────────────
_DEFAULTS = {
    "chat_history": [],
    "indexed": False,
    "chunks": [],
    "chunks_count": 0,
    "doc_name": "",
    "feedback": {},
    "strategy": RetrievalStrategy.HYBRID.value,
    "enable_rewrite": True,
    "enable_web_fallback": False,
}
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ── Cached heavy singletons ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_llm():
    return get_llm()


@st.cache_resource(show_spinner=False)
def _get_vectorstore():
    return get_vectorstore()


try:
    _llm = _get_llm()
    _vs = _get_vectorstore()
except Exception as exc:
    st.error(f"❌ **Initialisation failed:** {exc}")
    st.stop()


# ── Helper: latency bars ──────────────────────────────────────────────────────
def _latency_bars(latency: dict) -> str:
    total = max(latency.get("total_ms", 1), 1)
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


# ── Header ────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Document ingestion + settings
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📄 Load Document")
    st.caption("Upload a file or paste text to index it into Pinecone.")

    uploaded_file = st.file_uploader(
        "PDF / TXT / DOCX",
        type=["pdf", "txt", "docx"],
        label_visibility="collapsed",
    )
    input_text = st.text_area(
        "…or paste text", height=120,
        placeholder="Paste your document here",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Retrieval Settings")

    chosen_strategy = st.radio(
        "Strategy",
        options=[s.value for s in RetrievalStrategy],
        index=1,  # default: hybrid
        format_func=lambda s: {
            "dense":  "Dense (cosine)",
            "hybrid": "Hybrid (BM25 + RRF)",
            "mmr":    "MMR (diversity)",
        }[s],
    )
    # Always keep session state in sync with the widget
    st.session_state.strategy = chosen_strategy

    enable_rewrite = st.toggle(
        "Query rewriting (HyDE-lite)",
        value=st.session_state.enable_rewrite,
    )
    st.session_state.enable_rewrite = enable_rewrite

    enable_web_fallback = st.toggle(
        "Web search fallback",
        value=st.session_state.enable_web_fallback,
    )
    st.session_state.enable_web_fallback = enable_web_fallback

    st.markdown("---")

    # ── Index Document button ─────────────────────────────────────────────────
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

                    # Upload to Pinecone — pass the cached vectorstore
                    num = add_documents_to_vectorstore(chunks, _vs)
                    elapsed = time.perf_counter() - t0

                    # Persist chunks in session state so BM25 can use them
                    st.session_state.chunks = chunks
                    st.session_state.chunks_count = num
                    st.session_state.indexed = True
                    st.session_state.doc_name = (
                        uploaded_file.name if uploaded_file else "Pasted text"
                    )
                    st.session_state.chat_history = []  # fresh chat for new doc

                    st.success(f"✅ {num} chunks indexed in {elapsed:.2f}s")
                    st.rerun()   # ← force UI refresh so the chat area appears

                except Exception as exc:
                    st.error(f"❌ **Indexing failed:** {exc}")

    if st.session_state.indexed:
        st.info(
            f"📚 **{st.session_state.doc_name}**\n\n"
            f"{st.session_state.chunks_count} chunks · "
            f"{st.session_state.strategy} retrieval"
        )

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Chat interface
# ─────────────────────────────────────────────────────────────────────────────
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

# ── Render existing chat history ──────────────────────────────────────────────
for idx, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"], avatar="🧠" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            if msg.get("latency"):
                with st.expander("⏱ Latency breakdown", expanded=False):
                    st.markdown(_latency_bars(msg["latency"]), unsafe_allow_html=True)

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

            for w in msg.get("warnings", []):
                st.markdown(f'<div class="grounding-warn">{w}</div>', unsafe_allow_html=True)

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
                st.caption("✅ Thanks!" if current_fb == "up" else "📝 Noted.")


# ── Chat input ────────────────────────────────────────────────────────────────
query = st.chat_input("Ask a question about the document…")

if query:
    # Persist user message first so it shows immediately
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(query)

    # Read settings from session state (always current, even after sidebar rerun)
    strategy_enum = RetrievalStrategy(st.session_state.strategy)
    _enable_rewrite = st.session_state.enable_rewrite
    _enable_web_fallback = st.session_state.enable_web_fallback

    # Corpus for BM25 — must be a non-empty list for HYBRID to work
    corpus = st.session_state.chunks if st.session_state.chunks else None

    tracer = QueryTracer(query)

    with st.chat_message("assistant", avatar="🧠"):
        # ── Step 1: Build retriever ───────────────────────────────────────────
        retriever = create_retriever(_vs, corpus=corpus, strategy=strategy_enum)

        # ── Step 2: Retrieve docs ─────────────────────────────────────────────
        retrieval_query = query
        if _enable_rewrite:
            from rag_chain import rewrite_query
            with tracer.stage("embed"):
                retrieval_query = rewrite_query(query, _llm)

        with tracer.stage("retrieve"):
            retrieved_docs = retriever.invoke(retrieval_query)

        # Web fallback when Pinecone returns nothing useful
        if _enable_web_fallback:
            from web_search import augment_with_web
            retrieved_docs = augment_with_web(retrieved_docs, query)

        # ── Step 3: Build chain and stream answer ─────────────────────────────
        chain = create_rag_chain(retriever, _llm)

        if not retrieved_docs:
            # No context — tell the user clearly rather than wasting LLM tokens
            answer = (
                "⚠️ **No relevant chunks were retrieved from the document.** "
                "This usually means the document hasn't been indexed yet, or the "
                "question doesn't match the content. Try re-indexing the document."
            )
            st.markdown(answer)
        else:
            with tracer.stage("generate"):
                # Format context manually so streaming works cleanly
                from langchain_core.output_parsers import StrOutputParser
                from langchain_core.prompts import ChatPromptTemplate
                from rag_chain import RAG_PROMPT_TEMPLATE, format_docs

                prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
                stream_chain = prompt | _llm | StrOutputParser()
                context_str = format_docs(retrieved_docs)
                token_stream = stream_chain.stream(
                    {"context": context_str, "question": query}
                )
                answer = st.write_stream(token_stream)

        latency_dict = tracer.latency.as_dict()
        warnings = verify_citations(answer, len(retrieved_docs))

        # ── Display latency / sources / warnings ──────────────────────────────
        with st.expander("⏱ Latency breakdown", expanded=False):
            st.markdown(_latency_bars(latency_dict), unsafe_allow_html=True)

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
                    if "page" in doc.metadata:
                        st.caption(f"Page {doc.metadata['page']}")

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

    tracer.log(
        num_docs=len(retrieved_docs),
        strategy=st.session_state.strategy,
        query_rewritten=(retrieval_query != query),
    )
