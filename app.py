"""
Mini RAG - Streamlit UI Application
Main entry point for the RAG system.
"""

# IMPORTANT: Load .env FIRST before any other imports
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import time
# Load .env file explicitly
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path, override=True)



# Validate API keys BEFORE importing modules that use them
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "mini-rag")

# if not GOOGLE_API_KEY or not PINECONE_API_KEY:
#     st.set_page_config(page_title="Mini RAG - Setup Required", page_icon="⚠️")
#     st.error(
#         "❌ **Missing API Keys**\n\n"
#         "Please create a `.env` file in the project root with:\n\n"
#         "```\n"
#         "GOOGLE_API_KEY=your_google_key_here\n"
#         "PINECONE_API_KEY=your_pinecone_key_here\n"
#         "PINECONE_INDEX_NAME=mini-rag\n"
#         "```\n\n"
#         "**Get your keys:**\n"
#         "- Google: https://aistudio.google.com/\n"
#         "- Pinecone: https://www.pinecone.io/\n\n"
#         f"*Debug: .env exists: {env_path.exists()}*"
#     )
#     st.stop()


import streamlit as st

load_dotenv()

def get_secret(key, default=None):
    return st.secrets.get(key) or os.getenv(key) or default

GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
PINECONE_API_KEY = get_secret("PINECONE_API_KEY")
PINECONE_INDEX_NAME = get_secret("PINECONE_INDEX_NAME", "mini-rag")




if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    st.set_page_config(page_title="Setup Required", page_icon="⚠️")
    st.error("❌ Missing API keys. Configure Streamlit Secrets or .env")
    st.stop()



# Now import modular components (API keys are guaranteed to be available)
from config import Config
from embeddings import get_embeddings
from llm import get_llm
from vectorstore import get_vectorstore, add_documents_to_vectorstore
from retriever import create_retriever
from rag_chain import create_rag_chain, query_rag
from document_processor import process_documents


# Page config
st.set_page_config(
    page_title="Mini RAG App",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Mini RAG App - Gemini + Pinecone")
st.markdown("*Document ingestion, semantic search, and grounded generation with citations*")

# Initialize session state
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0

# Initialize components (lazy - only when needed)
@st.cache_resource
def get_components():
    """Load all components once and cache them."""
    try:
        return {
            "embeddings": get_embeddings(),
            "llm": get_llm(),
            "vectorstore": get_vectorstore(),
        }
    except Exception as e:
        st.error(f"❌ **Failed to initialize components:**\n\n{str(e)}")
        st.stop()

components = get_components()


# ============================================================================
# SIDEBAR: Document Ingestion
# ============================================================================

with st.sidebar:
    st.header("📄 Load Document")
    st.divider()
    
    # Input options
    input_text = st.text_area("Paste text here", height=150)
    uploaded_file = st.file_uploader(
        "Or upload PDF/TXT file",
        type=["pdf", "txt"]
    )
    
    # Index button
    if st.button("📤 Index Document", use_container_width=True, type="primary"):
        if not input_text and not uploaded_file:
            st.warning("Please provide text or upload a file.")
        else:
            with st.spinner("⏳ Processing & indexing..."):
                try:
                    start_time = time.time()
                    
                    # Process documents
                    chunks = process_documents(
                        uploaded_file=uploaded_file,
                        input_text=input_text
                    )
                    st.info(f"📊 Processed {len(chunks)} chunks")
                    
                    # Add to vectorstore
                    num_chunks = add_documents_to_vectorstore(
                        chunks,
                        components["vectorstore"]
                    )
                    
                    elapsed = time.time() - start_time
                    
                    # Update session state
                    st.session_state.chunks = chunks
                    st.session_state.indexed = True
                    st.session_state.chunks_count = num_chunks
                    
                    st.success(f" Successfully indexed {num_chunks} chunks in {elapsed:.2f}s")
                    
                except Exception as e:
                    error_msg = str(e)
                    st.error(
                        f"❌ **Error during indexing:**\n\n{error_msg}\n\n"
                        f"**Troubleshooting:**\n"
                        f"1. Make sure your Pinecone index exists\n"
                        f"2. Check PINECONE_API_KEY in .env\n"
                        f"3. Verify index name matches PINECONE_INDEX_NAME"
                    )
    
    # Status indicator
    if st.session_state.indexed:
        st.info(f"✓ {st.session_state.chunks_count} chunks indexed and ready for queries")


# ============================================================================
# MAIN: Query & Retrieval
# ============================================================================

if st.session_state.indexed:
    # Query input
    query = st.text_input(
        "🔍 Ask a question about the document",
        placeholder="What does the document say about...?"
    )
    
    if query:
        with st.spinner("⏳ Retrieving & generating..."):
            try:
                start_time = time.time()
                
                # Create retriever and chain
                retriever = create_retriever(components["vectorstore"])
                chain = create_rag_chain(retriever, components["llm"])
                
                # Query
                answer, retrieved_docs = query_rag(chain, retriever, query)
                
                elapsed = time.time() - start_time
                
                # Display answer
                st.markdown("### 📝 Answer")
                st.markdown(answer)
                st.caption(f"⏱️ Response time: {elapsed:.2f}s")
                
                # Display sources
                st.divider()
                st.markdown("### 📚 Sources")
                
                cols = st.columns([1, 3])
                for i, doc in enumerate(retrieved_docs):
                    with cols[0]:
                        st.badge(f"[{i+1}]")
                    with cols[1]:
                        st.markdown(f"**{doc.metadata.get('source', 'Unknown')}**")
                        st.text(doc.metadata.get('source_snippet', doc.page_content[:200]))
                        if "page" in doc.metadata:
                            st.caption(f"Page {doc.metadata['page']}")
                
            except Exception as e:
                st.error(f"❌ Error during retrieval: {str(e)}")

else:
    # Empty state
    st.info(
        "👈 **Get started:**\n\n"
        "1. Upload a PDF/TXT file or paste text in the sidebar\n"
        "2. Click 'Index Document'\n"
        "3. Ask questions about the content\n\n"
        "The app uses semantic search + reranking to find relevant passages, "
        "then generates grounded answers with citations."
    )
