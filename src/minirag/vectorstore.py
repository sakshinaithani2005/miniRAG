"""
Vector store module for miniRAG.
Handles Pinecone vector store initialization and document operations.
"""

from __future__ import annotations

import time

from .config import Config, get_settings
from .observability import get_logger
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def initialize_vectorstore() -> PineconeVectorStore:
    """
    Initialise a PineconeVectorStore against the configured index.

    Embeddings are resolved lazily here (after env vars are set) to avoid
    the module-import-before-env-var ordering bug.
    """
    from .embeddings import get_embeddings  # lazy — env vars already set by app.py

    pinecone_api_key = Config.get_pinecone_api_key()
    index_name = Config.get_pinecone_index_name() or "mini-rag"

    if not pinecone_api_key:
        raise ValueError(
            "PINECONE_API_KEY is not set. "
            "Add it to your .env file or Streamlit Secrets."
        )

    pc = Pinecone(api_key=pinecone_api_key)
    indexes = pc.list_indexes()
    index_names = [idx.name for idx in indexes.indexes]

    if index_name not in index_names:
        raise ValueError(
            f"Pinecone index '{index_name}' not found.\n"
            f"Available indexes: {index_names or 'None'}\n"
            "Create one at https://app.pinecone.io (dimension=3072, metric=cosine)"
        )

    return PineconeVectorStore(
        index_name=index_name,
        embedding=get_embeddings(),
        namespace="",  # default namespace
    )


def get_vectorstore() -> PineconeVectorStore:
    """Create and return a fresh PineconeVectorStore (no module-level singleton)."""
    return initialize_vectorstore()


# ---------------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------------

def add_documents_to_vectorstore(
    documents: list[Document],
    vectorstore: PineconeVectorStore | None = None,
) -> int:
    """
    Upload chunked documents to Pinecone.

    Clears the default namespace before uploading so each indexing run
    starts fresh. Documents must already have ``chunk_id`` and
    ``source_snippet`` in their metadata (set by ``process_documents``).

    Args:
        documents:   Chunked Documents from ``process_documents()``.
        vectorstore: An existing PineconeVectorStore.  A new one is created
                     if not supplied.

    Returns:
        Number of documents actually added.
    """
    if not documents:
        return 0

    if vectorstore is None:
        vectorstore = get_vectorstore()

    # ── Clear previous documents from default namespace ───────────────────────
    # We use the low-level Pinecone client to delete by namespace because
    # PineconeVectorStore.delete(delete_all=True) doesn't always respect the
    # namespace parameter correctly across versions.
    try:
        pinecone_api_key = Config.get_pinecone_api_key()
        index_name = Config.get_pinecone_index_name() or "mini-rag"
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(index_name)
        index.delete(delete_all=True, namespace="")
        # Brief pause so Pinecone propagates the delete before new upserts
        time.sleep(0.5)
    except Exception as exc:
        # 404 "Namespace not found" is expected on first run — safe to ignore
        logger.debug("Could not clear previous docs (likely first run)", exc=str(exc))

    # ── Upload in batches of 100 to avoid request-size limits ─────────────────
    BATCH = 100
    added = 0
    for i in range(0, len(documents), BATCH):
        batch = documents[i : i + BATCH]
        vectorstore.add_documents(batch)
        added += len(batch)

    return added


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def clear_vectorstore(vectorstore: PineconeVectorStore | None = None) -> None:
    """Delete all vectors from the default namespace."""
    try:
        pinecone_api_key = Config.get_pinecone_api_key()
        index_name = Config.get_pinecone_index_name() or "mini-rag"
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(index_name)
        index.delete(delete_all=True, namespace="")
    except Exception as exc:
        logger.warning("clear_vectorstore failed", error=str(exc))
