"""
Hybrid retrieval module for miniRAG.

Combines:
  - Dense retrieval  : Pinecone cosine similarity
  - Sparse retrieval : BM25 (term-frequency based)
  - Fusion           : Reciprocal Rank Fusion (RRF)

This pattern closes the vocabulary gap that pure dense search misses
(exact keyword matches, proper nouns, codes, IDs) while keeping the
semantic power of embedding-based retrieval.

Reference: Cormack, Clarke & Buettcher (2009) — Reciprocal Rank Fusion
"""

from __future__ import annotations

import math
from typing import List

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


# ---------------------------------------------------------------------------
# BM25 helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


class BM25Index:
    """Lightweight BM25 wrapper over a fixed corpus of Document objects."""

    def __init__(self, documents: List[Document]) -> None:
        self.documents = documents
        tokenized = [_tokenize(d.page_content) for d in documents]
        self._bm25 = BM25Okapi(tokenized)

    def get_scores(self, query: str) -> List[float]:
        """Return BM25 scores for all documents given a query string."""
        return list(self._bm25.get_scores(_tokenize(query)))

    def get_top_n(self, query: str, n: int = 10) -> List[Document]:
        """Return the top-n documents ranked by BM25 score."""
        scores = self.get_scores(query)
        ranked = sorted(zip(scores, self.documents), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:n]]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    k: int = 60,
) -> List[Document]:
    """
    Fuse multiple ranked document lists using Reciprocal Rank Fusion.

    RRF score = Σ  1 / (k + rank_i)   for each list i that contains the doc.

    Args:
        ranked_lists: Each inner list is a ranked list of Documents.
        k:            Smoothing constant (default 60 per the original paper).

    Returns:
        Single merged list of Documents ordered by descending RRF score.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            # Use page_content hash as a stable document key
            doc_id = _doc_key(doc)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            doc_map[doc_id] = doc

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_id] for doc_id, _ in fused]


def _doc_key(doc: Document) -> str:
    """Stable string key for a Document (chunk_id preferred, else hash of content)."""
    if "chunk_id" in doc.metadata:
        return str(doc.metadata["chunk_id"])
    return str(hash(doc.page_content[:200]))


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Combines dense (Pinecone) + sparse (BM25) retrieval via RRF.

    Usage::

        bm25_index = BM25Index(all_chunks)
        hybrid = HybridRetriever(vectorstore, bm25_index, top_k=10)
        docs = hybrid.retrieve("What is multi-head attention?")
    """

    def __init__(
        self,
        vectorstore,
        bm25_index: BM25Index,
        top_k: int = 10,
        rrf_k: int = 60,
    ) -> None:
        self._vs = vectorstore
        self._bm25 = bm25_index
        self._top_k = top_k
        self._rrf_k = rrf_k

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve documents using hybrid RRF over dense + sparse results.

        Args:
            query: Natural-language question string.

        Returns:
            Merged, deduplicated list of Documents ordered by RRF score.
        """
        # Dense retrieval (double the candidates to give RRF more to work with)
        dense_docs: List[Document] = self._vs.similarity_search(
            query, k=self._top_k * 2
        )

        # Sparse retrieval
        sparse_docs: List[Document] = self._bm25.get_top_n(query, n=self._top_k * 2)

        # Fuse
        fused = reciprocal_rank_fusion(
            [dense_docs, sparse_docs],
            k=self._rrf_k,
        )

        return fused[: self._top_k]

    def as_langchain_retriever(self):
        """
        Wrap as a LangChain-compatible retriever object.

        The returned object has an ``invoke(query)`` method so it can be
        used in LCEL chains.
        """
        return _LangChainRetrieverAdapter(self)


class _LangChainRetrieverAdapter:
    """Minimal adapter so HybridRetriever works in LCEL pipelines."""

    def __init__(self, hybrid: HybridRetriever) -> None:
        self._hybrid = hybrid

    def invoke(self, query: str) -> List[Document]:  # noqa: D102
        return self._hybrid.retrieve(query)

    def __or__(self, other):
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(self.invoke) | other
