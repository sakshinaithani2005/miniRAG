"""Tests for embeddings module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from minirag.embeddings import get_embeddings


def test_get_embeddings_without_api_key():
    with patch("minirag.config.Config.get_google_api_key", return_value=""):
        with pytest.raises(ValueError, match="GOOGLE_API_KEY is not set"):
            get_embeddings()


def test_get_embeddings_success():
    with patch("minirag.config.Config.get_google_api_key", return_value="fake-key"):
        with patch("minirag.embeddings.GoogleGenerativeAIEmbeddings") as mock_embeddings:
            get_embeddings()
            mock_embeddings.assert_called_once_with(
                model="gemini-embedding-001",
                task_type="RETRIEVAL_DOCUMENT",
                google_api_key="fake-key",
            )
