"""Tests for config module."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src" / "minirag"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC.parent.parent))


# ── RetrievalStrategy ──────────────────────────────────────────────────────────

def test_retrieval_strategy_values():
    from config import RetrievalStrategy

    assert RetrievalStrategy.DENSE.value == "dense"
    assert RetrievalStrategy.HYBRID.value == "hybrid"
    assert RetrievalStrategy.MMR.value == "mmr"


def test_retrieval_strategy_from_string():
    from config import RetrievalStrategy

    assert RetrievalStrategy("hybrid") == RetrievalStrategy.HYBRID


# ── Config shim ────────────────────────────────────────────────────────────────

def test_config_get_pinecone_index_name_default():
    from config import Config

    with patch.dict(os.environ, {"PINECONE_INDEX_NAME": "my-index"}):
        name = Config.get_pinecone_index_name()
    assert name == "my-index"


def test_config_validate_false_without_keys():
    from unittest.mock import patch as mk_patch

    from config import Config

    # Patch the classmethod helpers to return empty strings
    with mk_patch.object(Config, "get_google_api_key", return_value=""), \
         mk_patch.object(Config, "get_pinecone_api_key", return_value=""):
        result = Config.validate()
    assert result is False


# ── Settings ──────────────────────────────────────────────────────────────────

def test_settings_defaults():
    """Settings should have sensible defaults for non-secret fields."""
    from config import Settings

    with patch.dict(os.environ, {
        "GOOGLE_API_KEY": "fake_google",
        "PINECONE_API_KEY": "fake_pinecone",
    }):
        s = Settings()

    assert s.chunk_size == 1000
    assert s.chunk_overlap == 200
    assert s.rerank_top_n == 5
    assert s.llm_temperature == 0.3
    assert s.pinecone_index_name == "mini-rag"


def test_settings_override_via_env():
    from config import Settings

    with patch.dict(os.environ, {
        "GOOGLE_API_KEY": "g",
        "PINECONE_API_KEY": "p",
        "CHUNK_SIZE": "512",
        "LLM_TEMPERATURE": "0.7",
    }):
        s = Settings()

    assert s.chunk_size == 512
    assert s.llm_temperature == pytest.approx(0.7)
