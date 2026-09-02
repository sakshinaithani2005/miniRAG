"""Tests for web_search fallback module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from minirag.web_search import augment_with_web, should_fallback


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_docs(n, score=None):
    from langchain_core.documents import Document
    docs = []
    for i in range(n):
        meta = {"chunk_id": i + 1}
        if score is not None:
            meta["relevance_score"] = score
        docs.append(Document(page_content=f"doc {i}", metadata=meta))
    return docs


# ── should_fallback ────────────────────────────────────────────────────────────

def test_should_fallback_empty_docs():
    assert should_fallback([]) is True


def test_should_fallback_too_few_docs():
    docs = _make_docs(1)
    assert should_fallback(docs) is True


def test_should_fallback_low_score():
    docs = _make_docs(3, score=0.1)
    assert should_fallback(docs, threshold=0.30) is True


def test_should_fallback_high_score():
    docs = _make_docs(3, score=0.9)
    assert should_fallback(docs, threshold=0.30) is False


def test_should_fallback_no_score_enough_docs():
    docs = _make_docs(4)  # no scores in metadata
    assert should_fallback(docs) is False


# ── augment_with_web ──────────────────────────────────────────────────────────

def test_augment_no_fallback_when_scores_high():
    docs = _make_docs(3, score=0.95)
    result = augment_with_web(docs, "test query", threshold=0.30)
    assert len(result) == len(docs)  # unchanged


def test_augment_fallback_calls_web_search():
    mock_web_doc = MagicMock()
    with patch("minirag.web_search.web_search", return_value=[mock_web_doc]) as mock_ws:
        docs = _make_docs(1)  # triggers fallback (too few docs)
        result = augment_with_web(docs, "test query")
        mock_ws.assert_called_once()
        assert mock_web_doc in result
