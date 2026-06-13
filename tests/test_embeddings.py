"""Tests for embeddings module."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src" / "minirag"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from embeddings import get_embeddings


def test_get_embeddings_without_api_key():
    with patch("config.Config.get_google_api_key", return_value=""):
        with pytest.raises(ValueError, match="GOOGLE_API_KEY is not set"):
            get_embeddings()


def test_get_embeddings_success():
    with patch("config.Config.get_google_api_key", return_value="fake-key"):
        with patch("embeddings.GoogleGenerativeAIEmbeddings") as mock_embeddings:
            get_embeddings()
            mock_embeddings.assert_called_once_with(
                model="gemini-embedding-001",
                task_type="RETRIEVAL_DOCUMENT",
                google_api_key="fake-key",
            )
