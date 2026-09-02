"""Tests for hybrid_retriever module."""

from __future__ import annotations

from langchain_core.documents import Document

from minirag.hybrid_retriever import BM25Index, reciprocal_rank_fusion


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_docs(texts):
    return [Document(page_content=t, metadata={"chunk_id": i + 1}) for i, t in enumerate(texts)]


# ── BM25Index ─────────────────────────────────────────────────────────────────

def test_bm25_index_top_n_returns_correct_count():
    docs = _make_docs(["alpha beta gamma", "delta epsilon", "zeta eta theta"])
    idx = BM25Index(docs)
    results = idx.get_top_n("alpha", n=2)
    assert len(results) == 2


def test_bm25_index_scores_length():
    docs = _make_docs(["foo bar", "baz qux", "quux corge"])
    idx = BM25Index(docs)
    scores = idx.get_scores("foo")
    assert len(scores) == len(docs)


def test_bm25_index_keyword_match_ranks_first():
    docs = _make_docs([
        "transformers are neural networks",
        "attention mechanism in deep learning",
        "the quick brown fox",
    ])
    idx = BM25Index(docs)
    top = idx.get_top_n("attention mechanism", n=1)
    assert "attention" in top[0].page_content.lower()


# ── reciprocal_rank_fusion ────────────────────────────────────────────────────

def test_rrf_merges_lists():
    docs_a = _make_docs(["doc a1", "doc a2"])
    docs_b = _make_docs(["doc b1"])

    fused = reciprocal_rank_fusion([docs_a, docs_b])
    assert len(fused) >= 1


def test_rrf_deduplicates():
    # Same documents in both lists
    docs = _make_docs(["shared doc"])
    fused = reciprocal_rank_fusion([docs, docs])
    assert len(fused) == 1  # deduplicated


def test_rrf_higher_ranked_docs_score_higher():
    docs = _make_docs([f"doc {i}" for i in range(5)])
    # First doc is ranked #1 in both lists
    fused = reciprocal_rank_fusion([docs, docs])
    # The first document should remain at position 0
    assert fused[0].page_content == docs[0].page_content
