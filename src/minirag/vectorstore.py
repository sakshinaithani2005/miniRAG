"""
Vector store module for Mini RAG app.
Handles Pinecone vector store initialization and operations.
"""

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.documents import Document
from embeddings import get_embeddings
from config import Config
from typing import List
import os


def initialize_vectorstore() -> PineconeVectorStore:
    """
    Initialize Pinecone vector store.
    
    Returns:
        PineconeVectorStore instance
    """
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=Config.get_pinecone_api_key())
        index_name = Config.get_pinecone_index_name()
        
        # Verify index exists by listing available indexes
        indexes = pc.list_indexes()
        index_names = [idx.name for idx in indexes.indexes]
        
        if index_name not in index_names:
            raise ValueError(
                f"❌ Pinecone index '{index_name}' not found.\n"
                f"Available indexes: {index_names if index_names else 'None'}\n"
                f"Create an index at: https://app.pinecone.io"
            )
        
        # Create vectorstore - uses default namespace if not specified
        vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=get_embeddings(),
            namespace=""  # Use default namespace (empty string)
        )
        
        return vectorstore
        
    except Exception as e:
        raise ConnectionError(
            f"❌ Failed to initialize Pinecone: {str(e)}\n\n"
            f"Make sure:\n"
            f"1. PINECONE_API_KEY is correct\n"
            f"2. PINECONE_INDEX_NAME exists in Pinecone console\n"
            f"3. Your Pinecone account is active"
        )


# Singleton instance
_vectorstore_instance = None


def get_vectorstore() -> PineconeVectorStore:
    """
    Get or create vectorstore instance (singleton pattern).
    
    Returns:
        PineconeVectorStore instance
    """
    global _vectorstore_instance
    if _vectorstore_instance is None:
        _vectorstore_instance = initialize_vectorstore()
    return _vectorstore_instance


def add_documents_to_vectorstore(
    documents: List[Document],
    vectorstore: PineconeVectorStore = None
) -> int:
    """
    Add documents to vector store with metadata for citations.
    
    Args:
        documents: List of Document objects to add
        vectorstore: PineconeVectorStore instance (uses singleton if None)
    
    Returns:
        Number of documents added
    """
    if vectorstore is None:
        vectorstore = get_vectorstore()
    
    # Add metadata for citation
    for i, doc in enumerate(documents):
        doc.metadata.update({
            "chunk_id": i + 1,
            "source_snippet": doc.page_content[:Config.SNIPPET_LENGTH] + "..."
        })
    
    try:
        # Try to clear previous documents (may fail if namespace is empty)
        vectorstore.delete(delete_all=True)
    except Exception as e:
        # If delete fails, it might be the first time - that's okay
        print(f"Note: Could not clear previous docs (likely first time): {str(e)}")
    
    # Add new documents
    vectorstore.add_documents(documents)
    
    return len(documents)


def clear_vectorstore(vectorstore: PineconeVectorStore = None) -> None:
    """
    Clear all documents from vector store.
    
    Args:
        vectorstore: PineconeVectorStore instance (uses singleton if None)
    """
    if vectorstore is None:
        vectorstore = get_vectorstore()
    
    vectorstore.delete(delete_all=True)
