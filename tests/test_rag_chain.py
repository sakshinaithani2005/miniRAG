"""Tests for rag_chain module."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.documents import Document

from minirag.rag_chain import format_docs, query_rag, rewrite_query, verify_citations


# ── format_docs ───────────────────────────────────────────────────────────────

def test_format_docs_numbering():
    docs = [
        Document(page_content="Alpha text", metadata={"source": "doc.pdf", "chunk_id": 1}),
        Document(page_content="Beta text",  metadata={"source": "doc.pdf", "chunk_id": 2}),
    ]
    result = format_docs(docs)
    assert "[1]" in result
    assert "[2]" in result
    assert "Alpha text" in result
    assert "Beta text" in result


def test_format_docs_source_metadata():
    docs = [Document(page_content="text", metadata={"source": "paper.pdf", "chunk_id": 5})]
    result = format_docs(docs)
    assert "paper.pdf" in result
    assert "5" in result


def test_format_docs_missing_metadata():
    docs = [Document(page_content="text", metadata={})]
    result = format_docs(docs)
    # Should not raise — falls back to "Unknown"
    assert "Unknown" in result
    assert "?" in result


# ── verify_citations ──────────────────────────────────────────────────────────

def test_verify_citations_valid():
    answer = "Attention is all you need [1]. Multi-head attention [2] is key."
    warnings = verify_citations(answer, num_docs=3)
    assert warnings == []


def test_verify_citations_hallucinated():
    answer = "This cites a non-existent source [5]."
    warnings = verify_citations(answer, num_docs=3)
    assert len(warnings) == 1
    assert "5" in warnings[0]


def test_verify_citations_no_citations():
    answer = "No citations in this answer."
    warnings = verify_citations(answer, num_docs=5)
    assert warnings == []


def test_verify_citations_multiple_hallucinated():
    answer = "Bad refs [7][8][99]."
    warnings = verify_citations(answer, num_docs=3)
    assert len(warnings) == 1  # single warning lists all
    assert "7" in warnings[0] or "8" in warnings[0]


# ── rewrite_query ─────────────────────────────────────────────────────────────

def test_rewrite_query_falls_back_on_error():
    mock_llm = MagicMock()
    mock_llm.__or__ = MagicMock(side_effect=Exception("LLM unavailable"))

    # Should return original question, not raise
    result = rewrite_query("What is BERT?", mock_llm)
    assert isinstance(result, str)
    assert len(result) > 0


# ── query_rag ─────────────────────────────────────────────────────────────────

def test_query_rag_returns_tuple():
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Answer with [1] citation."

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="Context", metadata={"chunk_id": 1, "source": "doc"})
    ]

    answer, docs, warnings = query_rag(
        mock_chain,
        mock_retriever,
        "test question",
        enable_rewrite=False,
    )

    assert isinstance(answer, str)
    assert isinstance(docs, list)
    assert isinstance(warnings, list)
    assert "Answer" in answer
