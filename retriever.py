"""
Retriever module for Mini RAG app.
Handles retrieval and reranking logic.
"""

from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_pinecone import PineconeVectorStore
from config import Config


def initialize_reranker() -> FlashrankRerank:
    """
    Initialize FlashRank reranker.
    
    Returns:
        FlashrankRerank instance
    """
    try:
        reranker = FlashrankRerank(top_n=Config.RERANK_TOP_N)
        # Rebuild pydantic model to ensure full definition (Pydantic v2 compatibility)
        if hasattr(reranker, 'model_rebuild'):
            reranker.model_rebuild()
        return reranker
    except Exception as e:
        # If FlashrankRerank fails, create a simple wrapper that just returns top k docs
        print(f"⚠️  FlashrankRerank initialization warning: {str(e)}")
        print("   Using basic retrieval without reranking")
        return None


# Singleton instance
_reranker_instance = None


def get_reranker() -> FlashrankRerank:
    """
    Get or create reranker instance (singleton pattern).
    
    Returns:
        FlashrankRerank instance (or None if initialization failed)
    """
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = initialize_reranker()
    return _reranker_instance


def create_retriever(vectorstore: PineconeVectorStore) -> ContextualCompressionRetriever:
    """
    Create a compression retriever with FlashRank reranking.
    
    Args:
        vectorstore: PineconeVectorStore instance
    
    Returns:
        ContextualCompressionRetriever instance
    """
    # Base retriever from vectorstore
    base_retriever = vectorstore.as_retriever(
        search_kwargs={"k": Config.RETRIEVAL_TOP_K}
    )
    
    # Try to use compression retriever with FlashRank
    reranker = get_reranker()
    
    if reranker is not None:
        # Compression retriever with FlashRank
        try:
            retriever = ContextualCompressionRetriever(
                base_compressor=reranker,
                base_retriever=base_retriever
            )
            return retriever
        except Exception as e:
            print(f"⚠️  Failed to create compression retriever: {str(e)}")
            print("   Falling back to basic retriever")
            return base_retriever
    else:
        # If reranker failed, just return base retriever
        return base_retriever
