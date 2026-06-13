"""
Document processing module for miniRAG.

Handles document loading, chunking, and text extraction.
Supports: PDF, TXT, DOCX, and raw text strings.
"""

from __future__ import annotations

import hashlib
import os
import tempfile

from config import Config
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# File-hash deduplication helper
# ---------------------------------------------------------------------------

def compute_file_hash(content: bytes) -> str:
    """Return a short SHA-256 hex digest for deduplication keying."""
    return hashlib.sha256(content).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_document_from_file(uploaded_file) -> list[Document]:
    """
    Load a document from a Streamlit uploaded file object.

    Supports PDF, TXT, and DOCX.  A ``file_hash`` field is added to every
    document's metadata so duplicate uploads can be detected and each document
    can be stored in its own Pinecone namespace.

    Args:
        uploaded_file: ``streamlit.UploadedFile`` object.

    Returns:
        List of ``Document`` objects.
    """
    raw_bytes: bytes = uploaded_file.getvalue()
    file_hash = compute_file_hash(raw_bytes)
    mime_type: str = getattr(uploaded_file, "type", "") or ""
    file_name: str = getattr(uploaded_file, "name", "upload")

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=_suffix(mime_type, file_name)
    ) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        if mime_type == "application/pdf" or file_name.lower().endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif file_name.lower().endswith(".docx"):
            try:
                from langchain_community.document_loaders import Docx2txtLoader  # noqa
                loader = Docx2txtLoader(tmp_path)
            except ImportError as e:
                raise ImportError(
                    "DOCX support requires 'docx2txt'. Install with: pip install docx2txt"
                ) from e
        else:
            loader = TextLoader(tmp_path, autodetect_encoding=True)

        documents = loader.load()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    for doc in documents:
        doc.metadata.setdefault("source", file_name)
        doc.metadata["file_hash"] = file_hash

    return documents


def _suffix(mime_type: str, file_name: str) -> str:
    if mime_type == "application/pdf":
        return ".pdf"
    if file_name.lower().endswith(".docx"):
        return ".docx"
    return ".txt"


def load_text_from_string(
    text: str, source_name: str = "pasted_text"
) -> list[Document]:
    """
    Wrap a raw text string as a single Document.

    Args:
        text:        Raw text content.
        source_name: Label used in the ``source`` metadata field.

    Returns:
        List containing a single ``Document``.
    """
    file_hash = compute_file_hash(text.encode())
    return [
        Document(
            page_content=text,
            metadata={"source": source_name, "file_hash": file_hash},
        )
    ]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Split documents into overlapping chunks.

    Args:
        documents:     Input documents.
        chunk_size:    Override ``Config.CHUNK_SIZE`` when provided.
        chunk_overlap: Override ``Config.CHUNK_OVERLAP`` when provided.

    Returns:
        List of chunked ``Document`` objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE if chunk_size is None else chunk_size,
        chunk_overlap=Config.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap,
        separators=Config.CHUNK_SEPARATORS,
    )
    return splitter.split_documents(documents)


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------

def process_documents(
    uploaded_file=None,
    input_text: str | None = None,
) -> list[Document]:
    """
    Complete document processing pipeline: load → chunk → enrich metadata.

    Args:
        uploaded_file: Optional Streamlit ``UploadedFile``.
        input_text:    Optional raw text string.

    Returns:
        List of chunked, metadata-enriched ``Document`` objects.

    Raises:
        ValueError: If neither argument is supplied.
    """
    if not uploaded_file and not input_text:
        raise ValueError("Either uploaded_file or input_text must be provided.")

    documents = (
        load_document_from_file(uploaded_file)
        if uploaded_file
        else load_text_from_string(input_text)
    )

    chunks = chunk_documents(documents)

    # Stamp each chunk with a sequential ID and snippet
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i + 1
        snippet = chunk.page_content[: Config.SNIPPET_LENGTH]
        chunk.metadata["source_snippet"] = (
            snippet + "..."
            if len(chunk.page_content) > Config.SNIPPET_LENGTH
            else snippet
        )

    return chunks
