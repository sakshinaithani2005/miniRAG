"""
LLM module for Mini RAG app.
Handles Google Gemini LLM initialization.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from config import Config


def initialize_llm() -> ChatGoogleGenerativeAI:
    """
    Initialize Google Gemini LLM.
    
    Returns:
        ChatGoogleGenerativeAI instance
    """
    return ChatGoogleGenerativeAI(
        model=Config.LLM_MODEL,
        temperature=Config.LLM_TEMPERATURE,
        google_api_key=Config.get_google_api_key()
    )


# Singleton instance
_llm_instance = None


def get_llm() -> ChatGoogleGenerativeAI:
    """
    Get or create LLM instance (singleton pattern).
    
    Returns:
        ChatGoogleGenerativeAI instance
    """
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = initialize_llm()
    return _llm_instance
