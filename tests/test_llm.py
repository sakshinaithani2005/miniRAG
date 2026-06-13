"""Tests for llm module."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src" / "minirag"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llm import get_llm


def test_get_llm_without_api_key():
    with patch("config.Config.get_google_api_key", return_value=""):
        with pytest.raises(ValueError, match="GOOGLE_API_KEY is not set"):
            get_llm()


def test_get_llm_success():
    with patch("config.Config.get_google_api_key", return_value="fake-key"):
        with patch("llm.ChatGoogleGenerativeAI") as mock_llm:
            get_llm()
            mock_llm.assert_called_once_with(
                model="gemini-2.5-flash",
                temperature=0.3,
                google_api_key="fake-key",
            )
