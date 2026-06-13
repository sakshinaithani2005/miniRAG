"""
Configuration module for miniRAG.

Uses pydantic-settings as the single source of truth.
Values are loaded from (in priority order):
  1. Environment variables
  2. .env file in project root
  3. Streamlit secrets (when running under Streamlit — lazy import)
  4. Hardcoded defaults

NOTE: streamlit is imported lazily so this module works in test/CLI
      contexts where Streamlit is not available.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/minirag/ -> repo root


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies."""

    DENSE = "dense"    # Pinecone cosine similarity only
    HYBRID = "hybrid"  # Dense + BM25 with Reciprocal Rank Fusion
    MMR = "mmr"        # Maximal Marginal Relevance (diversity-aware)


class Settings(BaseSettings):
    """All application configuration in one place."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API Keys ──────────────────────────────────────────────────────────────
    google_api_key: str = Field(default="", description="Google AI Studio API key")
    pinecone_api_key: str = Field(default="", description="Pinecone API key")
    pinecone_index_name: str = Field("mini-rag", description="Pinecone index name")

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_model: str = Field("gemini-embedding-001")
    embedding_dimension: int = Field(3072)
    embedding_task_type: str = Field("RETRIEVAL_DOCUMENT")

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_model: str = Field("gemini-2.5-flash")
    llm_temperature: float = Field(0.3, ge=0.0, le=2.0)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(10, ge=1)
    rerank_top_n: int = Field(5, ge=1)
    retrieval_strategy: RetrievalStrategy = Field(RetrievalStrategy.HYBRID)
    low_score_threshold: float = Field(0.30)

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = Field(1000, ge=100)
    chunk_overlap: int = Field(200, ge=0)
    chunk_separators: list[str] = Field(default=["\n\n", "\n", " ", ""])
    snippet_length: int = Field(200, ge=50)

    # ── Feature flags ─────────────────────────────────────────────────────────
    enable_query_rewriting: bool = Field(True)
    enable_web_fallback: bool = Field(False)

    # ── Observability ─────────────────────────────────────────────────────────
    log_level: str = Field("INFO")
    enable_langsmith_tracing: bool = Field(False)
    langsmith_api_key: Optional[str] = Field(None)
    langsmith_project: str = Field("miniRAG")

    @model_validator(mode="after")
    def _apply_streamlit_secrets(self) -> "Settings":
        """Pull any missing values from st.secrets (Streamlit Cloud)."""
        try:
            import streamlit as st  # type: ignore  # noqa: PLC0415

            for field_name in ("google_api_key", "pinecone_api_key", "pinecone_index_name"):
                if not getattr(self, field_name, None):
                    env_name = field_name.upper()
                    if hasattr(st, "secrets") and env_name in st.secrets:
                        object.__setattr__(self, field_name, st.secrets[env_name])
        except Exception:
            # Not running under Streamlit — silently skip
            pass
        return self

    @model_validator(mode="after")
    def _configure_langsmith(self) -> "Settings":
        if self.enable_langsmith_tracing and self.langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project
        return self


def get_settings() -> Settings:
    """Return a Settings instance."""
    return Settings()


# ---------------------------------------------------------------------------
# Legacy Config shim — keeps old imports working
# ---------------------------------------------------------------------------

class Config:
    """Backward-compatible shim. New code should use ``get_settings()``."""

    _s: Optional[Settings] = None

    @classmethod
    def _settings(cls) -> Settings:
        if cls._s is None:
            cls._s = get_settings()
        return cls._s

    EMBEDDING_MODEL = "gemini-embedding-001"
    EMBEDDING_DIMENSION = 3072
    EMBEDDING_TASK_TYPE = "RETRIEVAL_DOCUMENT"
    LLM_MODEL = "gemini-2.5-flash"
    LLM_TEMPERATURE = 0.3
    RETRIEVAL_TOP_K = 10
    RERANK_TOP_N = 5
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    CHUNK_SEPARATORS = ["\n\n", "\n", " ", ""]
    SNIPPET_LENGTH = 200

    @classmethod
    def get_google_api_key(cls) -> Optional[str]:
        try:
            return cls._settings().google_api_key or os.getenv("GOOGLE_API_KEY")
        except Exception:
            return os.getenv("GOOGLE_API_KEY")

    @classmethod
    def get_pinecone_api_key(cls) -> Optional[str]:
        try:
            return cls._settings().pinecone_api_key or os.getenv("PINECONE_API_KEY")
        except Exception:
            return os.getenv("PINECONE_API_KEY")

    @classmethod
    def get_pinecone_index_name(cls) -> Optional[str]:
        try:
            return cls._settings().pinecone_index_name or os.getenv("PINECONE_INDEX_NAME", "mini-rag")
        except Exception:
            return os.getenv("PINECONE_INDEX_NAME", "mini-rag")

    @classmethod
    def validate(cls) -> bool:
        try:
            s = cls._settings()
            return bool(s.google_api_key and s.pinecone_api_key and s.pinecone_index_name)
        except Exception as exc:
            print(f"⚠️  Config validation failed: {exc}")
            # Fall back to env var check
            return bool(os.getenv("GOOGLE_API_KEY") and os.getenv("PINECONE_API_KEY"))
