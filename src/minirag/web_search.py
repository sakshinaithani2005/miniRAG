"""
Web search fallback module for miniRAG.

When retrieved context scores are below the configured threshold, this
module automatically augments the answer with a real-time DuckDuckGo
search result — making miniRAG an agentic, adaptive RAG system.

Requires: ``duckduckgo-search`` (pip install duckduckgo-search)
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document


def web_search(query: str, max_results: int = 3) -> List[Document]:
    """
    Run a DuckDuckGo search and return results as Document objects.

    Args:
        query:       Search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of ``Document`` objects with search result snippets.
        Returns an empty list if ``duckduckgo-search`` is not installed.
    """
    try:
        from duckduckgo_search import DDGS  # type: ignore
    except ImportError:
        print("⚠️  duckduckgo-search not installed. Run: pip install duckduckgo-search")
        return []

    results: List[Document] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    Document(
                        page_content=f"{r.get('title', '')}\n{r.get('body', '')}",
                        metadata={
                            "source": r.get("href", "web"),
                            "source_type": "web_search",
                            "chunk_id": f"web-{len(results) + 1}",
                        },
                    )
                )
    except Exception as exc:
        print(f"⚠️  DuckDuckGo search failed: {exc}")

    return results


def should_fallback(retrieved_docs: List[Document], threshold: float = 0.30) -> bool:
    """
    Decide whether to trigger the web-search fallback.

    Heuristic: fall back if fewer than 2 docs were retrieved OR if the
    average relevance score (when present in metadata) is below the threshold.

    Args:
        retrieved_docs: Docs returned by the primary retriever.
        threshold:      Minimum average score before triggering fallback.

    Returns:
        ``True`` if web search should be used to augment context.
    """
    if len(retrieved_docs) < 2:
        return True

    # Collect whichever score metadata key is present per document
    scores = []
    for doc in retrieved_docs:
        score = doc.metadata.get("relevance_score")
        if score is None:
            score = doc.metadata.get("score")
        if score is not None:
            scores.append(float(score))

    if scores:
        avg_score = sum(scores) / len(scores)
        return avg_score < threshold

    # No score metadata — only trigger if very few docs returned
    return len(retrieved_docs) < 2


def augment_with_web(
    retrieved_docs: List[Document],
    query: str,
    max_results: int = 3,
    threshold: float = 0.30,
) -> List[Document]:
    """
    Optionally augment retrieved docs with web search results.

    Checks ``should_fallback`` and appends web results when triggered.
    Web results are clearly labelled with ``source_type: web_search``.

    Args:
        retrieved_docs: Primary retrieval results.
        query:          User query string.
        max_results:    Max web results to fetch.
        threshold:      Score threshold for triggering fallback.

    Returns:
        Combined list (original docs first, web docs appended).
    """
    if not should_fallback(retrieved_docs, threshold):
        return retrieved_docs

    print(f"🌐 Context score below threshold — augmenting with web search: '{query}'")
    web_docs = web_search(query, max_results=max_results)
    return retrieved_docs + web_docs
