"""Tests for vectorstore module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from minirag.vectorstore import (
    add_documents_to_vectorstore,
    clear_vectorstore,
    get_vectorstore,
    initialize_vectorstore,
)


def test_initialize_vectorstore_without_api_key():
    with patch("minirag.config.Config.get_pinecone_api_key", return_value=""):
        with pytest.raises(ValueError, match="PINECONE_API_KEY is not set"):
            initialize_vectorstore()


def test_initialize_vectorstore_index_not_found():
    with patch("minirag.config.Config.get_pinecone_api_key", return_value="fake-pinecone-key"):
        with patch("minirag.config.Config.get_pinecone_index_name", return_value="my-index"):
            with patch("minirag.vectorstore.Pinecone") as mock_pinecone:
                mock_pc = MagicMock()
                mock_idx = MagicMock()
                mock_idx.name = "other-index"
                mock_pc.list_indexes.return_value.indexes = [mock_idx]
                mock_pinecone.return_value = mock_pc

                with pytest.raises(ValueError, match="Pinecone index 'my-index' not found"):
                    initialize_vectorstore()


def test_initialize_vectorstore_success():
    with patch("minirag.config.Config.get_pinecone_api_key", return_value="fake-pinecone-key"):
        with patch("minirag.config.Config.get_pinecone_index_name", return_value="my-index"):
            with patch("minirag.vectorstore.Pinecone") as mock_pinecone:
                mock_pc = MagicMock()
                mock_idx = MagicMock()
                mock_idx.name = "my-index"
                mock_pc.list_indexes.return_value.indexes = [mock_idx]
                mock_pinecone.return_value = mock_pc

                with patch("minirag.embeddings.get_embeddings") as mock_get_embed:
                    mock_get_embed.return_value = MagicMock()
                    with patch("minirag.vectorstore.PineconeVectorStore") as mock_vs_cls:
                        initialize_vectorstore()
                        mock_vs_cls.assert_called_once_with(
                            index_name="my-index",
                            embedding=mock_get_embed.return_value,
                            namespace="",
                        )


def test_get_vectorstore():
    with patch("minirag.vectorstore.initialize_vectorstore") as mock_init:
        get_vectorstore()
        mock_init.assert_called_once()


def test_add_documents_to_vectorstore_empty():
    assert add_documents_to_vectorstore([]) == 0


def test_add_documents_to_vectorstore_success():
    docs = [Document(page_content=f"chunk {i}") for i in range(150)]
    mock_vs = MagicMock()

    with patch("minirag.config.Config.get_pinecone_api_key", return_value="fake-pinecone-key"):
        with patch("minirag.config.Config.get_pinecone_index_name", return_value="my-index"):
            with patch("minirag.vectorstore.Pinecone") as mock_pinecone:
                mock_pc = MagicMock()
                mock_index = MagicMock()
                mock_pc.Index.return_value = mock_index
                mock_pinecone.return_value = mock_pc

                added = add_documents_to_vectorstore(docs, mock_vs)

                mock_pc.Index.assert_called_once_with("my-index")
                mock_index.delete.assert_called_once_with(delete_all=True, namespace="")

                assert mock_vs.add_documents.call_count == 2
                assert added == 150


def test_clear_vectorstore():
    with patch("minirag.config.Config.get_pinecone_api_key", return_value="fake-pinecone-key"):
        with patch("minirag.config.Config.get_pinecone_index_name", return_value="my-index"):
            with patch("minirag.vectorstore.Pinecone") as mock_pinecone:
                mock_pc = MagicMock()
                mock_index = MagicMock()
                mock_pc.Index.return_value = mock_index
                mock_pinecone.return_value = mock_pc

                clear_vectorstore(MagicMock())

                mock_pc.Index.assert_called_once_with("my-index")
                mock_index.delete.assert_called_once_with(delete_all=True, namespace="")
