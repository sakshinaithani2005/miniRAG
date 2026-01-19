"""
Configuration module for Mini RAG app.
Handles environment variables and API keys.
"""

import os
import streamlit as st
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path

# Load .env file from project root
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path) 



def get_api_key(key_name: str) -> Optional[str]:
    """
    Retrieve API key from Streamlit secrets, environment variables, or .env file.
    
    Args:
        key_name: Name of the key (e.g., 'GOOGLE_API_KEY')
    
    Returns:
        API key string or None if not found
    """
    # Try Streamlit secrets first
    try:
        if hasattr(st, 'secrets') and key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    
    # Fall back to environment variables
    return os.getenv(key_name)


class Config:
    """Configuration class for Mini RAG app."""
    
    # Embeddings config
    EMBEDDING_MODEL = "gemini-embedding-001"
    EMBEDDING_DIMENSION = 3072
    EMBEDDING_TASK_TYPE = "RETRIEVAL_DOCUMENT"
    
    # LLM config
    LLM_MODEL = "gemini-2.5-flash"
    LLM_TEMPERATURE = 0.3
    
    # Retrieval config
    RETRIEVAL_TOP_K = 10  # Initial retrieval
    RERANK_TOP_N = 5      # After reranking
    
    # Chunking config
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    CHUNK_SEPARATORS = ["\n\n", "\n", " ", ""]
    SNIPPET_LENGTH = 200  # Characters for source snippet
    
    @classmethod
    def get_google_api_key(cls) -> Optional[str]:
        """Get Google API key from environment or secrets."""
        return os.getenv("GOOGLE_API_KEY") or (st.secrets.get("GOOGLE_API_KEY") if hasattr(st, 'secrets') else None)
    
    @classmethod
    def get_pinecone_api_key(cls) -> Optional[str]:
        """Get Pinecone API key from environment or secrets."""
        return os.getenv("PINECONE_API_KEY") or (st.secrets.get("PINECONE_API_KEY") if hasattr(st, 'secrets') else None)
    
    @classmethod
    def get_pinecone_index_name(cls) -> Optional[str]:
        """Get Pinecone index name from environment or secrets."""
        return os.getenv("PINECONE_INDEX_NAME") or (st.secrets.get("PINECONE_INDEX_NAME") if hasattr(st, 'secrets') else "mini-rag")
    
    # For backward compatibility - these are properties that fetch fresh values
    @property
    def GOOGLE_API_KEY(self) -> Optional[str]:
        return self.get_google_api_key()
    
    @property
    def PINECONE_API_KEY(self) -> Optional[str]:
        return self.get_pinecone_api_key()
    
    @property
    def PINECONE_INDEX_NAME(self) -> Optional[str]:
        return self.get_pinecone_index_name()
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate that all required API keys are set.
        
        Returns:
            True if all keys are present, False otherwise
        """
        # Get fresh values from environment
        google_key = cls.get_google_api_key()
        pinecone_key = cls.get_pinecone_api_key()
        pinecone_index = cls.get_pinecone_index_name()
        
        required_keys = [
            ("GOOGLE_API_KEY", google_key),
            ("PINECONE_API_KEY", pinecone_key),
            ("PINECONE_INDEX_NAME", pinecone_index),
        ]
        
        missing = [key_name for key_name, value in required_keys if not value]
        
        if missing:
            print(f"⚠️  Missing API keys: {', '.join(missing)}")
            print(f"DEBUG: .env file exists: {env_path.exists()}")
            if env_path.exists():
                print(f"DEBUG: GOOGLE_API_KEY in env: {'GOOGLE_API_KEY' in os.environ}")
                print(f"DEBUG: PINECONE_API_KEY in env: {'PINECONE_API_KEY' in os.environ}")
            return False
        
        return True
