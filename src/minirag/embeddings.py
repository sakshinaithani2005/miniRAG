"""
Embeddings module for Mini RAG app.
Handles Google Gemini embeddings initialization.
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import Config


def initialize_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Initialize Google Gemini embeddings.
    
    Returns:
        GoogleGenerativeAIEmbeddings instance
    """
    return GoogleGenerativeAIEmbeddings(
        model=Config.EMBEDDING_MODEL,
        task_type=Config.EMBEDDING_TASK_TYPE,
        google_api_key=Config.get_google_api_key()
    )


# Singleton instance
_embeddings_instance = None


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Get or create embeddings instance (singleton pattern).
    
    Returns:
        GoogleGenerativeAIEmbeddings instance
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = initialize_embeddings()
    return _embeddings_instance
