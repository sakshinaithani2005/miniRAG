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
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env", override=True)

import streamlit as st

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="miniRAG · Gemini + Pinecone",
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
.tag-crag {
    background: rgba(236,72,153,0.2);
    border: 1px solid rgba(236,72,153,0.5);
    color: #f472b6;
}
.tag-grade {
    background: rgba(59,130,246,0.2);
    border: 1px solid rgba(59,130,246,0.5);
    color: #93c5fd;
}
.tag-grounded {
    background: rgba(34,197,94,0.2);
    border: 1px solid rgba(34,197,94,0.5);
    color: #86efac;
}
.crag-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(236,72,153,0.3);
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
    margin: 0.5rem 0;
    font-size: 0.82rem;
}
.latency-bar-wrap { margin: 0.5rem 0 1rem; }
.latency-label { font-size: 0.72rem; color: #a78bfa; margin-bottom: 4px; }
.latency-bar {
    height: 6px;
    border-radius: 4px;
    background: linear-gradient(90deg, #8b5cf6, #ec4899);
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
        "**Missing API keys.**\n\n"
        "Add `GOOGLE_API_KEY` and `PINECONE_API_KEY` to your `.env` file or Streamlit Secrets."
    )
    st.stop()

# Use os.environ[] (not setdefault) so keys are always current
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["PINECONE_INDEX_NAME"] = PINECONE_INDEX_NAME

# ── Lazy imports (after env vars are set) ────────────────────────────────────
from minirag import (
    CRAGPipeline,
    RAG_PROMPT_TEMPLATE,
    RetrievalStrategy,
    add_documents_to_vectorstore,
    augment_with_web,
    configure_logging,
    create_retriever,
    format_docs,
    get_llm,
    get_vectorstore,
    process_documents,
    rewrite_query,
    verify_citations,
)
from minirag.observability import QueryTracer
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

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
    "enable_crag": True,
    "enable_grounding": True,
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
    st.error(f"**Initialization failed:** {exc}")
    st.stop()


# ── Helper: latency bars ──────────────────────────────────────────────────────
def _latency_bars(latency: dict) -> str:
    total = max(latency.get("total_ms", 1), 1)
    rows = ""
    for stage, label in [
        ("embed_ms",    "Rewrite"),
        ("retrieve_ms", "Retrieve"),
        ("grade_ms",    "Grade Docs"),
        ("rerank_ms",   "Rerank"),
        ("generate_ms", "Generate"),
        ("ground_ms",   "Grounding"),
    ]:
        ms = latency.get(stage, 0)
        if ms > 0:
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
        "<h1 style='margin-bottom:0;color:#e2d9f3;'>miniRAG</h1>"
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
    st.markdown("### Load Document")
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
    st.markdown("### Retrieval Settings")

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

    enable_crag = st.toggle(
        "Corrective RAG (CRAG Agent)",
        value=st.session_state.enable_crag,
        help="Self-reflective agent: grades document relevance, reformulates queries, and triggers smart fallback.",
    )
    st.session_state.enable_crag = enable_crag

    enable_grounding = st.toggle(
        "Grounding & Hallucination Guardrail",
        value=st.session_state.enable_grounding,
        help="Automated post-generation self-critique verifying context faithfulness.",
    )
    st.session_state.enable_grounding = enable_grounding

    enable_web_fallback = st.toggle(
        "Web search fallback (Standard RAG)",
        value=st.session_state.enable_web_fallback,
        disabled=st.session_state.enable_crag,
        help="Static threshold web fallback (disabled when CRAG is active as CRAG manages fallback dynamically).",
    )
    st.session_state.enable_web_fallback = enable_web_fallback

    st.markdown("---")

    # ── Index Document button ─────────────────────────────────────────────────
    if st.button("Index Document", use_container_width=True, type="primary"):
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

                    st.success(f"{num} chunks indexed in {elapsed:.2f}s")
                    st.rerun()   # ← force UI refresh so the chat area appears

                except Exception as exc:
                    st.error(f"**Indexing failed:** {exc}")

    if st.session_state.indexed:
        st.info(
            f"**{st.session_state.doc_name}**\n\n"
            f"{st.session_state.chunks_count} chunks · "
            f"{st.session_state.strategy} retrieval"
        )

    st.markdown("---")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Chat interface
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.indexed:
    st.markdown(
        '<div class="glass-card">'
        "<h3 style='color:#e2d9f3;'>Welcome to miniRAG</h3>"
        "<p style='color:#9ca3af;'>Upload a PDF, TXT, or DOCX in the sidebar — "
        "or paste raw text — then ask questions and get grounded, cited answers.</p>"
        "<ul style='color:#6b7280;'>"
        "<li><b>Hybrid BM25 + dense retrieval</b> for best coverage</li>"
        "<li><b>FlashRank reranking</b> for precision</li>"
        "<li><b>Query rewriting</b> for ambiguous questions</li>"
        "<li><b>RAGAS evaluation</b> — run <code>python eval/eval_pipeline.py</code></li>"
        "</ul>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ── Render existing chat history ──────────────────────────────────────────────
for idx, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("crag_action"):
            action = msg["crag_action"]
            action_map = {
                "CORRECT_DOCS_DIRECT": ("Direct Context", "tag-grounded"),
                "AMBIGUOUS_WEB_AUGMENTED": ("Web-Augmented", "tag-crag"),
                "INSUFFICIENT_WEB_FALLBACK": ("Web Fallback", "tag-web"),
            }
            label, tag_class = action_map.get(action, ("CRAG", "tag-crag"))
            ratio = msg.get("relevance_ratio", 1.0)
            st.markdown(
                f'<span class="tag {tag_class}">{label}</span> '
                f'<span class="tag tag-grade">Relevance: {ratio:.0%}</span>',
                unsafe_allow_html=True,
            )

        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            if msg.get("latency"):
                with st.expander("Latency breakdown", expanded=False):
                    st.markdown(_latency_bars(msg["latency"]), unsafe_allow_html=True)

            if msg.get("grounding_summary"):
                st.caption(f"**Grounding Audit:** {msg['grounding_summary']}")

            if msg.get("sources"):
                with st.expander(f"Sources ({len(msg['sources'])} chunks)", expanded=False):
                    for i, doc in enumerate(msg["sources"], 1):
                        source_type = doc.metadata.get("source_type", "")
                        tag_cls = "tag-web" if source_type == "web_search" else "tag"
                        label = "Web" if source_type == "web_search" else f"[{i}]"
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
            c1, c2, _ = st.columns([1.2, 1.4, 7.4])
            with c1:
                if st.button("Helpful", key=f"up_{idx}", disabled=current_fb is not None):
                    st.session_state.feedback[idx] = "up"
                    st.rerun()
            with c2:
                if st.button("Unhelpful", key=f"dn_{idx}", disabled=current_fb is not None):
                    st.session_state.feedback[idx] = "down"
                    st.rerun()
            if current_fb:
                st.caption("Thank you for your feedback!" if current_fb == "up" else "Feedback recorded.")


# ── Chat input ────────────────────────────────────────────────────────────────
query = st.chat_input("Ask a question about the document…")

if query:
    # Persist user message first so it shows immediately
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Read settings from session state
    strategy_enum = RetrievalStrategy(st.session_state.strategy)
    _enable_rewrite = st.session_state.enable_rewrite
    _enable_crag = st.session_state.enable_crag
    _enable_grounding = st.session_state.enable_grounding
    _enable_web_fallback = st.session_state.enable_web_fallback

    corpus = st.session_state.chunks if st.session_state.chunks else None
    tracer = QueryTracer(query)

    with st.chat_message("assistant"):
        retriever = create_retriever(_vs, corpus=corpus, strategy=strategy_enum)

        crag_action = None
        relevance_ratio = 1.0
        grounding_summary = None

        if _enable_crag:
            crag = CRAGPipeline(
                retriever=retriever,
                llm=_llm,
                relevance_threshold=0.5,
                enable_grounding_check=_enable_grounding,
            )

            (
                final_docs,
                grading_result,
                crag_action,
                transformed_query,
                web_fallback_used,
            ) = crag.prepare_context(query, tracer=tracer)

            relevance_ratio = grading_result.relevance_ratio

            action_map = {
                "CORRECT_DOCS_DIRECT": ("Direct Context (High Precision)", "tag-grounded"),
                "AMBIGUOUS_WEB_AUGMENTED": ("Web-Augmented (Partial Context)", "tag-crag"),
                "INSUFFICIENT_WEB_FALLBACK": ("Web Fallback (Low Retrieval Recall)", "tag-web"),
            }
            label, tag_class = action_map.get(crag_action, ("CRAG Active", "tag-crag"))
            st.markdown(
                f'<span class="tag {tag_class}">{label}</span> '
                f'<span class="tag tag-grade">Doc Relevance: {relevance_ratio:.0%}</span>',
                unsafe_allow_html=True,
            )

            if transformed_query:
                st.caption(f"**Query Reformulation:** `{transformed_query}`")

            if not final_docs:
                answer = "**No relevant chunks found in document or external search.**"
                st.markdown(answer)
            else:
                with tracer.stage("generate"):
                    token_stream = crag.stream_answer(query, final_docs)
                    answer = st.write_stream(token_stream)

            retrieved_docs = final_docs

            if _enable_grounding and final_docs:
                with tracer.stage("ground"):
                    report = crag.grounding_checker.check(answer, final_docs, query)
                    grounding_summary = report.summary
                    if not report.is_grounded and report.hallucinated_claims:
                        st.markdown(
                            f'<div class="grounding-warn">**Grounding Audit ({report.faithfulness_score:.0%}):** '
                            f'Unsupported claim(s): {", ".join(report.hallucinated_claims)}</div>',
                            unsafe_allow_html=True,
                        )

        else:
            # ── Standard Classic RAG ──────────────────────────────────────────
            retrieval_query = query
            if _enable_rewrite:
                with tracer.stage("embed"):
                    retrieval_query = rewrite_query(query, _llm)

            with tracer.stage("retrieve"):
                retrieved_docs = retriever.invoke(retrieval_query)

            if _enable_web_fallback:
                retrieved_docs = augment_with_web(retrieved_docs, query)

            if not retrieved_docs:
                answer = (
                    "**No relevant chunks were retrieved from the document.** "
                    "Try re-indexing or rephrasing your question."
                )
                st.markdown(answer)
            else:
                with tracer.stage("generate"):
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
        with st.expander("Latency breakdown", expanded=False):
            st.markdown(_latency_bars(latency_dict), unsafe_allow_html=True)

        if retrieved_docs:
            with st.expander(f"Sources ({len(retrieved_docs)} chunks)", expanded=False):
                for i, doc in enumerate(retrieved_docs, 1):
                    source_type = doc.metadata.get("source_type", "")
                    tag_cls = "tag-web" if source_type == "web_search" else "tag"
                    label = "Web" if source_type == "web_search" else f"[{i}]"
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
        c1, c2, _ = st.columns([1.2, 1.4, 7.4])
        with c1:
            st.button("Helpful", key=f"up_{msg_idx}")
        with c2:
            st.button("Unhelpful", key=f"dn_{msg_idx}")

    # Persist assistant message
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": retrieved_docs,
        "latency": latency_dict,
        "warnings": warnings,
        "crag_action": crag_action,
        "relevance_ratio": relevance_ratio,
        "grounding_summary": grounding_summary,
    })

    tracer.log(
        num_docs=len(retrieved_docs),
        strategy=st.session_state.strategy,
        crag_enabled=_enable_crag,
        crag_action=crag_action,
    )
