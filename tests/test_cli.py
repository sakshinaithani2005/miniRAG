"""Tests for cli module."""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.documents import Document

from minirag.cli import _parse_args, main


def test_parse_args():
    with patch(
        "sys.argv",
        ["cli.py", "--query", "what is this", "--strategy", "dense"],
    ):
        args = _parse_args()
        assert args.query == "what is this"
        assert args.strategy == "dense"
        assert not args.no_rewrite


def test_main_missing_keys():
    with patch("sys.argv", ["cli.py", "--query", "what"]):
        with patch("minirag.config.Config.validate", return_value=False):
            res = main()
            assert res == 1


def test_main_success_no_doc():
    with patch(
        "sys.argv",
        ["cli.py", "--query", "what", "--strategy", "dense", "--no-rewrite"],
    ):
        with patch("minirag.config.Config.validate", return_value=True):
            with patch("minirag.vectorstore.get_vectorstore"):
                with patch("minirag.retriever.create_retriever"):
                    with patch("minirag.llm.get_llm"):
                        with patch("minirag.rag_chain.create_rag_chain"):
                            with patch("minirag.rag_chain.query_rag") as mock_query:
                                mock_docs = [
                                    Document(
                                        page_content="some content",
                                        metadata={
                                            "source": "doc.txt",
                                            "chunk_id": 1,
                                        },
                                    )
                                ]
                                mock_query.return_value = (
                                    "RAG answer",
                                    mock_docs,
                                    [],
                                )

                                res = main()

                                assert res == 0
                                mock_query.assert_called_once()
