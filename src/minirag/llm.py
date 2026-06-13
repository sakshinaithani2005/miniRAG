"""
LLM module for miniRAG.
Handles Google Gemini LLM initialisation.
"""

from __future__ import annotations

from config import Config
from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm() -> ChatGoogleGenerativeAI:
    """
    Create a fresh ChatGoogleGenerativeAI instance.

    Called lazily after env vars are propagated so the API key is always
    current.  Streamlit caches the result via @st.cache_resource.
    """
    api_key = Config.get_google_api_key()
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. "
            "Add it to your .env file or Streamlit Secrets."
        )
    return ChatGoogleGenerativeAI(
        model=Config.LLM_MODEL,
        temperature=Config.LLM_TEMPERATURE,
        google_api_key=api_key,
    )
