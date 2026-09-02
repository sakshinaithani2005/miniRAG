"""Tests for document_processor module."""

from __future__ import annotations

import pytest

from minirag.config import Config
from minirag.document_processor import (
    chunk_documents,
    compute_file_hash,
    load_text_from_string,
    process_documents,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

class FakeUploadedFile:
    """Minimal Streamlit UploadedFile mock."""

    def __init__(self, content: bytes, name: str, mime: str):
        self._content = content
        self.name = name
        self.type = mime

    def getvalue(self) -> bytes:
        return self._content


# ── compute_file_hash ─────────────────────────────────────────────────────────

def test_compute_file_hash_stable():
    h1 = compute_file_hash(b"hello world")
    h2 = compute_file_hash(b"hello world")
    assert h1 == h2


def test_compute_file_hash_different_inputs():
    h1 = compute_file_hash(b"hello")
    h2 = compute_file_hash(b"world")
    assert h1 != h2


def test_compute_file_hash_length():
    h = compute_file_hash(b"test")
    assert len(h) == 16  # truncated to 16 hex chars


# ── load_text_from_string ─────────────────────────────────────────────────────

def test_load_text_from_string_basic():
    docs = load_text_from_string("Hello, world!")
    assert len(docs) == 1
    assert docs[0].page_content == "Hello, world!"


def test_load_text_from_string_metadata():
    docs = load_text_from_string("test", source_name="my_source")
    assert docs[0].metadata["source"] == "my_source"
    assert "file_hash" in docs[0].metadata


# ── chunk_documents ───────────────────────────────────────────────────────────

def test_chunk_documents_splits_long_text():
    from langchain_core.documents import Document

    long_text = "word " * 600  # > 1000 chars
    docs = [Document(page_content=long_text, metadata={"source": "test"})]
    chunks = chunk_documents(docs)
    assert len(chunks) > 1


def test_chunk_documents_preserves_metadata():
    from langchain_core.documents import Document

    doc = Document(page_content="short text", metadata={"source": "test", "file_hash": "abc123"})
    chunks = chunk_documents([doc])
    assert all(c.metadata.get("source") == "test" for c in chunks)


# ── process_documents ─────────────────────────────────────────────────────────

def test_process_documents_raises_without_input():
    with pytest.raises(ValueError, match="Either uploaded_file or input_text"):
        process_documents()


def test_process_documents_text_input():
    chunks = process_documents(input_text="Hello RAG world. " * 100)
    assert len(chunks) >= 1
    assert all("chunk_id" in c.metadata for c in chunks)
    assert all("source_snippet" in c.metadata for c in chunks)


def test_process_documents_chunk_ids_sequential():
    chunks = process_documents(input_text="test " * 300)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert ids == list(range(1, len(chunks) + 1))


def test_process_documents_snippet_length():
    chunks = process_documents(input_text="x " * 500)
    for c in chunks:
        snippet = c.metadata["source_snippet"]
        # snippet body (before "...") <= SNIPPET_LENGTH
        assert len(snippet.removesuffix("...")) <= Config.SNIPPET_LENGTH + 10  # small tolerance
