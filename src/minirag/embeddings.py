"""
Embeddings module for miniRAG.
Handles Google Gemini embedding initialisation.
"""

from __future__ import annotations

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import Config


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Create a fresh GoogleGenerativeAIEmbeddings instance.

    Called lazily after env vars are propagated so the API key is always
    current.  Streamlit caches the result via @st.cache_resource.

    Uses RETRIEVAL_DOCUMENT task type for both indexing and querying —
    this is correct for gemini-embedding-001 which handles both internally.
    """
    api_key = Config.get_google_api_key()
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. "
            "Add it to your .env file or Streamlit Secrets."
        )
    return GoogleGenerativeAIEmbeddings(
        model=Config.EMBEDDING_MODEL,
        task_type=Config.EMBEDDING_TASK_TYPE,
        google_api_key=api_key,
    )
