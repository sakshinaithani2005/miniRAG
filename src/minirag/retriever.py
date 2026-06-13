"""
Retriever module for miniRAG.

Supports three retrieval strategies (controlled by Config / Settings):
  - DENSE   : Pinecone cosine similarity + FlashRank reranking (original)
  - HYBRID  : Dense + BM25 via Reciprocal Rank Fusion + FlashRank reranking
  - MMR     : Maximal Marginal Relevance for diversity-aware retrieval

The active strategy is selected by the ``strategy`` argument to
``create_retriever()``; it defaults to ``RetrievalStrategy.HYBRID``.
"""

from __future__ import annotations

from config import Config, RetrievalStrategy
from hybrid_retriever import BM25Index, HybridRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

# ---------------------------------------------------------------------------
# Reranker singleton
# ---------------------------------------------------------------------------

_reranker_instance: FlashrankRerank | None = None


def get_reranker() -> FlashrankRerank | None:
    """
    Return a cached FlashrankRerank instance.

    Returns ``None`` if FlashRank fails to initialise; callers fall back
    to unranked results gracefully.
    """
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = _init_reranker()
    return _reranker_instance


def _init_reranker() -> FlashrankRerank | None:
    try:
        reranker = FlashrankRerank(top_n=Config.RERANK_TOP_N)
        if hasattr(reranker, "model_rebuild"):
            reranker.model_rebuild()
        return reranker
    except Exception as exc:
        print(f"⚠️  FlashrankRerank init warning: {exc} — falling back to unranked retrieval.")
        return None


# ---------------------------------------------------------------------------
# Strategy: DENSE  (Pinecone cosine + FlashRank)
# ---------------------------------------------------------------------------

def _dense_retriever(vectorstore: PineconeVectorStore, top_k: int):
    """Standard dense retrieval with optional FlashRank reranking."""
    base = vectorstore.as_retriever(search_kwargs={"k": top_k})
    reranker = get_reranker()
    if reranker is None:
        return base
    try:
        from langchain_classic.retrievers import ContextualCompressionRetriever  # noqa
        return ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base,
        )
    except Exception as exc:
        print(f"⚠️  ContextualCompressionRetriever failed: {exc} — using base retriever.")
        return base


# ---------------------------------------------------------------------------
# Strategy: MMR
# ---------------------------------------------------------------------------

def _mmr_retriever(vectorstore: PineconeVectorStore, top_k: int):
    """MMR retriever — promotes diversity in results."""
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": top_k * 3, "lambda_mult": 0.5},
    )


# ---------------------------------------------------------------------------
# Strategy: HYBRID  (BM25 + dense + RRF + FlashRank)
# ---------------------------------------------------------------------------

class _HybridWithRerank:
    """
    LangChain-compatible retriever that combines HybridRetriever + FlashRank.

    Exposes ``.invoke(query) -> List[Document]`` so it drops into LCEL chains.
    """

    def __init__(
        self,
        hybrid: HybridRetriever,
        reranker: FlashrankRerank | None,
        top_n: int,
    ) -> None:
        self._hybrid = hybrid
        self._reranker = reranker
        self._top_n = top_n

    def invoke(self, query: str) -> list[Document]:
        docs = self._hybrid.retrieve(query)
        if self._reranker is None:
            return docs[: self._top_n]
        try:
            compressed = self._reranker.compress_documents(docs, query)
            return list(compressed)[: self._top_n]
        except Exception:
            return docs[: self._top_n]

    # Allow use as the left-hand side of an LCEL `|` pipe
    def __or__(self, other):
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(self.invoke) | other


def _hybrid_retriever(
    vectorstore: PineconeVectorStore,
    corpus: list[Document],
    top_k: int,
) -> _HybridWithRerank:
    """Build a hybrid (dense + BM25 + RRF) retriever over the given corpus."""
    bm25_index = BM25Index(corpus)
    hybrid = HybridRetriever(vectorstore, bm25_index, top_k=top_k)
    return _HybridWithRerank(hybrid, get_reranker(), Config.RERANK_TOP_N)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def create_retriever(
    vectorstore: PineconeVectorStore,
    corpus: list[Document] | None = None,
    strategy: RetrievalStrategy | None = None,
):
    """
    Build a retriever according to the chosen retrieval strategy.

    Args:
        vectorstore: Initialised ``PineconeVectorStore``.
        corpus:      All indexed chunks — **required** for ``HYBRID`` strategy
                     (used to build the BM25 index over the same documents).
                     Pass ``None`` or an empty list to fall back to DENSE.
        strategy:    Which retrieval strategy to use.  Defaults to
                     ``RetrievalStrategy.HYBRID`` when corpus is available,
                     otherwise ``RetrievalStrategy.DENSE``.

    Returns:
        A retriever with an ``.invoke(query) -> List[Document]`` interface
        that is compatible with LangChain LCEL chains.
    """
    top_k = Config.RETRIEVAL_TOP_K
    chosen = strategy or RetrievalStrategy.HYBRID

    # Cannot do HYBRID without a corpus — fall back gracefully
    if chosen == RetrievalStrategy.HYBRID and not corpus:
        print("⚠️  HYBRID requested but no corpus provided — falling back to DENSE.")
        chosen = RetrievalStrategy.DENSE

    if chosen == RetrievalStrategy.MMR:
        return _mmr_retriever(vectorstore, top_k)
    elif chosen == RetrievalStrategy.HYBRID:
        return _hybrid_retriever(vectorstore, corpus, top_k)
    else:
        return _dense_retriever(vectorstore, top_k)
