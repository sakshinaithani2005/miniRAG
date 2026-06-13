"""Tests for cli module."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

_SRC = Path(__file__).resolve().parents[1] / "src" / "minirag"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cli import _parse_args, main


def test_parse_args():
    with patch("sys.argv", ["cli.py", "--query", "what is this", "--strategy", "dense"]):
        args = _parse_args()
        assert args.query == "what is this"
        assert args.strategy == "dense"
        assert not args.no_rewrite


def test_main_missing_keys():
    with patch("sys.argv", ["cli.py", "--query", "what"]):
        with patch("config.Config.validate", return_value=False):
            res = main()
            assert res == 1


def test_main_success_no_doc():
    with patch("sys.argv", ["cli.py", "--query", "what", "--strategy", "dense", "--no-rewrite"]):
        with patch("config.Config.validate", return_value=True):
            with patch("vectorstore.get_vectorstore") as mock_get_vs:
                with patch("retriever.create_retriever") as mock_create_retriever:
                    with patch("llm.get_llm") as mock_get_llm:
                        with patch("rag_chain.create_rag_chain") as mock_create_chain:
                            with patch("rag_chain.query_rag") as mock_query:
                                mock_docs = [
                                    Document(
                                        page_content="some content",
                                        metadata={"source": "doc.txt", "chunk_id": 1},
                                    )
                                ]
                                mock_query.return_value = ("RAG answer", mock_docs, [])

                                res = main()

                                assert res == 0
                                mock_query.assert_called_once()
