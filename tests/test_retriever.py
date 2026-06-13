"""Tests for retriever module."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

_SRC = Path(__file__).resolve().parents[1] / "src" / "minirag"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import RetrievalStrategy
from retriever import (
    _init_reranker,
    create_retriever,
    get_reranker,
)


def test_init_reranker_success():
    with patch("retriever.FlashrankRerank") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        res = _init_reranker()
        assert res == mock_instance
        mock_instance.model_rebuild.assert_called_once()


def test_init_reranker_failure():
    with patch(
        "retriever.FlashrankRerank", side_effect=Exception("error")
    ):
        res = _init_reranker()
        assert res is None


def test_get_reranker():
    with patch("retriever._reranker_instance", None):
        with patch("retriever._init_reranker") as mock_init:
            mock_init.return_value = MagicMock()
            r1 = get_reranker()
            r2 = get_reranker()
            assert r1 == r2
            mock_init.assert_called_once()


def test_create_retriever_mmr():
    mock_vs = MagicMock()
    create_retriever(mock_vs, strategy=RetrievalStrategy.MMR)
    mock_vs.as_retriever.assert_called_once_with(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.5},
    )


def test_create_retriever_dense_no_reranker():
    mock_vs = MagicMock()
    with patch("retriever.get_reranker", return_value=None):
        create_retriever(mock_vs, strategy=RetrievalStrategy.DENSE)
        mock_vs.as_retriever.assert_called_once_with(
            search_kwargs={"k": 10}
        )


def test_create_retriever_dense_with_reranker():
    mock_vs = MagicMock()
    mock_vs.as_retriever.return_value = MagicMock()
    mock_reranker = MagicMock()

    with patch("retriever.get_reranker", return_value=mock_reranker):
        with patch(
            "langchain_classic.retrievers.ContextualCompressionRetriever"
        ) as mock_ccr:
            create_retriever(mock_vs, strategy=RetrievalStrategy.DENSE)
            mock_ccr.assert_called_once()


def test_create_retriever_hybrid_no_corpus_fallback():
    mock_vs = MagicMock()
    with patch("retriever.get_reranker", return_value=None):
        create_retriever(
            mock_vs, corpus=None, strategy=RetrievalStrategy.HYBRID
        )
        mock_vs.as_retriever.assert_called_once_with(
            search_kwargs={"k": 10}
        )


def test_create_retriever_hybrid_success():
    mock_vs = MagicMock()
    corpus = [Document(page_content="hello world")]
    with patch("retriever.BM25Index") as mock_bm25:
        with patch("retriever.HybridRetriever") as mock_hybrid:
            with patch("retriever.get_reranker", return_value=None):
                retriever = create_retriever(
                    mock_vs, corpus=corpus, strategy=RetrievalStrategy.HYBRID
                )
                mock_bm25.assert_called_once_with(corpus)
                mock_hybrid.assert_called_once_with(
                    mock_vs, mock_bm25.return_value, top_k=10
                )
                assert retriever is not None


def test_hybrid_with_rerank_invoke():
    from retriever import _HybridWithRerank

    mock_hybrid = MagicMock()
    mock_docs = [
        Document(page_content="doc 1"),
        Document(page_content="doc 2"),
    ]
    mock_hybrid.retrieve.return_value = mock_docs

    r = _HybridWithRerank(mock_hybrid, None, top_n=1)
    res = r.invoke("query")
    assert res == mock_docs[:1]

    mock_reranker = MagicMock()
    mock_reranker.compress_documents.return_value = mock_docs[::-1]
    r2 = _HybridWithRerank(mock_hybrid, mock_reranker, top_n=2)
    res2 = r2.invoke("query")
    assert res2 == mock_docs[::-1]


def test_hybrid_with_rerank_pipe():
    from retriever import _HybridWithRerank

    r = _HybridWithRerank(MagicMock(), None, top_n=1)
    pipe = r | MagicMock()
    assert pipe is not None
