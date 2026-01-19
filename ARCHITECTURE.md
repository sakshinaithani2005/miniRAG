"""
Project Structure and Architecture Documentation

Module Breakdown:
================

config.py
  - Configuration class with all app parameters
  - API key management (secrets/env vars)
  - Centralized settings for chunking, retrieval, LLM, etc.
  - validate() method to check required keys

embeddings.py
  - Google Gemini embeddings initialization
  - Singleton pattern for embeddings instance
  - get_embeddings() for lazy loading

llm.py
  - Google Gemini LLM (gemini-1.5-flash) initialization
  - Singleton pattern for LLM instance
  - get_llm() for lazy loading

vectorstore.py
  - Pinecone vector store initialization
  - add_documents_to_vectorstore() - with metadata for citations
  - clear_vectorstore() - cleanup
  - Singleton pattern for vectorstore instance
  - Manages document upsert with chunk_id and source_snippet

retriever.py
  - FlashRank reranker initialization
  - create_retriever() - builds ContextualCompressionRetriever
  - Combines base retriever (top-10) + reranker (top-5)
  - Singleton pattern for reranker

document_processor.py
  - load_document_from_file() - PDF/TXT file handling
  - load_text_from_string() - pasted text handling
  - chunk_documents() - recursive text splitting with overlap
  - process_documents() - orchestrates full pipeline

rag_chain.py
  - RAG_PROMPT_TEMPLATE - system prompt with citation instructions
  - format_docs() - formats retrieved docs with numbering
  - create_rag_chain() - builds LangChain runnable chain
  - query_rag() - executes query and returns answer + sources

app.py (Streamlit UI)
  - Page configuration and setup
  - Config validation with error handling
  - Session state management
  - Sidebar: document upload/indexing UI
  - Main: query input and result display
  - Caching with @st.cache_resource
  - Clean, component-driven architecture


Data Flow:
==========

INGESTION:
  User Input (text/file)
    ↓
  process_documents()
    ├─ load_document_from_file() or load_text_from_string()
    ├─ chunk_documents()
    ↓
  add_documents_to_vectorstore()
    ├─ Add metadata (chunk_id, source_snippet)
    ├─ Clear old vectors
    ├─ Upsert to Pinecone
    ↓
  Session state updated

RETRIEVAL:
  User Query
    ↓
  create_retriever()
    ├─ Base retriever (k=10)
    ├─ FlashRank compressor (top_n=5)
    ↓
  create_rag_chain()
    ├─ Format docs with numbers
    ├─ Populate prompt with context
    ├─ LLM generates answer
    ↓
  query_rag()
    ├─ Get answer
    ├─ Get source documents
    ↓
  Display (answer + citations + sources)


Benefits of Modular Design:
===========================

1. Testability
   - Each module can be unit tested independently
   - Mock components easily for testing

2. Reusability
   - Modules can be used in other projects
   - Easy to swap implementations (e.g., Cohere reranker)

3. Maintainability
   - Clear separation of concerns
   - Changes to one module don't affect others
   - Easy to locate and fix bugs

4. Scalability
   - Add new features without modifying core
   - Easy to add caching, logging, monitoring
   - Can be converted to async if needed

5. Configuration
   - Centralized config management
   - Easy to add environment-specific settings
   - Different models/parameters per environment

6. Performance
   - Lazy loading with singletons
   - Streamlit caching with @st.cache_resource
   - Efficient resource management
"""
