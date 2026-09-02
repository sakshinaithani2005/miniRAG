"""Tests for llm module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from minirag.llm import get_llm


def test_get_llm_without_api_key():
    with patch("minirag.config.Config.get_google_api_key", return_value=""):
        with pytest.raises(ValueError, match="GOOGLE_API_KEY is not set"):
            get_llm()


def test_get_llm_success():
    with patch("minirag.config.Config.get_google_api_key", return_value="fake-key"):
        with patch("minirag.llm.ChatGoogleGenerativeAI") as mock_llm:
            get_llm()
            mock_llm.assert_called_once_with(
                model="gemini-2.5-flash",
                temperature=0.3,
                google_api_key="fake-key",
            )
