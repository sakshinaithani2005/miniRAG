"""
miniRAG — A production-grade RAG system using Gemini + Pinecone + LangChain.
"""

from .config import Config, RetrievalStrategy, Settings, get_settings
from .crag_pipeline import CRAGOutput, CRAGPipeline
from .document_processor import (
    chunk_documents,
    compute_file_hash,
    load_document_from_file,
    load_text_from_string,
    process_documents,
)
from .embeddings import get_embeddings
from .grader import DocumentGrader, GradedDocument, GradingResult
from .grounding import GroundingChecker, GroundingReport
from .hybrid_retriever import BM25Index, HybridRetriever, reciprocal_rank_fusion
from .llm import get_llm
from .observability import LatencyBreakdown, QueryTracer, configure_logging, get_logger
from .query_transform import QueryTransformer
from .rag_chain import create_rag_chain, format_docs, query_rag, rewrite_query, stream_rag, verify_citations
from .retriever import create_retriever, get_reranker
from .vectorstore import (
    add_documents_to_vectorstore,
    clear_vectorstore,
    get_vectorstore,
    initialize_vectorstore,
)
from .web_search import augment_with_web, should_fallback, web_search

__version__ = "0.2.0"
__author__ = "miniRAG contributors"

__all__ = [
    "Config",
    "Settings",
    "get_settings",
    "RetrievalStrategy",
    "CRAGPipeline",
    "CRAGOutput",
    "DocumentGrader",
    "GradedDocument",
    "GradingResult",
    "GroundingChecker",
    "GroundingReport",
    "QueryTransformer",
    "BM25Index",
    "HybridRetriever",
    "reciprocal_rank_fusion",
    "create_retriever",
    "get_reranker",
    "process_documents",
    "chunk_documents",
    "load_document_from_file",
    "load_text_from_string",
    "compute_file_hash",
    "initialize_vectorstore",
    "get_vectorstore",
    "add_documents_to_vectorstore",
    "clear_vectorstore",
    "get_embeddings",
    "get_llm",
    "create_rag_chain",
    "query_rag",
    "stream_rag",
    "rewrite_query",
    "verify_citations",
    "format_docs",
    "web_search",
    "augment_with_web",
    "should_fallback",
    "QueryTracer",
    "LatencyBreakdown",
    "configure_logging",
    "get_logger",
]
