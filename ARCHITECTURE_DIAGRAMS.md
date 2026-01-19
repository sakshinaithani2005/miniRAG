"""
Mini RAG - Modular Architecture Visualization

This document provides ASCII diagrams and dependency graphs for the modular system.
"""

## Dependency Graph

```
┌─────────────────────────────────────────────────────────┐
│                    app.py (UI Layer)                     │
│           Streamlit interface & orchestration            │
└────────────────────┬────────────────────────────────────┘
                     │
      ┌──────────────┼──────────────┐
      │              │              │
      ▼              ▼              ▼
┌──────────┐ ┌────────────┐ ┌──────────────┐
│ config.py│ │document_   │ │ retriever.py │
│          │ │processor.py│ │              │
└────┬─────┘ └──────┬──────┘ └──────┬───────┘
     │              │                │
     │         ┌────┴────┐      ┌────┴──────┐
     │         ▼         ▼      ▼           ▼
     │   ┌─────────────────────────────────────┐
     │   │    vectorstore.py                    │
     │   │    rag_chain.py                      │
     │   │    embeddings.py                     │
     │   │    llm.py                            │
     │   └──────────────┬──────────────────────┘
     │                  │
     │         ┌────────┴────────┐
     │         ▼                 ▼
     └─────→ External APIs:
             - Gemini Embeddings
             - Gemini LLM
             - Pinecone Vector DB
             - FlashRank Reranker
```

## Module Dependency Chain

```
config.py
  ↑ (imported by all modules)
  │
embeddings.py    llm.py    vectorstore.py    retriever.py
  │               │              │                │
  └───────────────┼──────────────┴────────────────┘
                  │
         document_processor.py    rag_chain.py
                  │                   │
                  └───────────┬───────┘
                              │
                         app.py (UI)
```

## Data Flow - Document Ingestion

```
User Action (Upload/Paste)
         │
         ▼
   ┌──────────────────────┐
   │  app.py              │
   │  (UI - Sidebar)      │
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────────────┐
   │ document_processor.py         │
   │ ├─ load_document_from_file()  │
   │ │  or                         │
   │ └─ load_text_from_string()    │
   └──────────┬───────────────────┘
              │
         Documents
              │
              ▼
   ┌──────────────────────────────┐
   │ document_processor.py         │
   │ ├─ chunk_documents()          │
   │ │  RecursiveCharacterText...  │
   │ │  Split (1000 chars, 200 OL) │
   └──────────┬───────────────────┘
              │
         Chunks
              │
              ▼
   ┌──────────────────────────────┐
   │ vectorstore.py               │
   │ ├─ add_documents_to_...()    │
   │ │  Add metadata (chunk_id)   │
   │ │  Clear old vectors         │
   │ │  Embed with Gemini         │
   │ │  Upsert to Pinecone        │
   └──────────┬───────────────────┘
              │
              ▼
   ┌──────────────────────────────┐
   │ Pinecone Vector Store        │
   │ 768-dimensional              │
   │ Cosine similarity            │
   └──────────────────────────────┘
```

## Data Flow - Query Execution

```
User Question
         │
         ▼
   ┌──────────────────┐
   │  app.py          │
   │  (UI - Query)    │
   └────────┬─────────┘
            │
            ▼
   ┌────────────────────────────────┐
   │ rag_chain.py                    │
   │ ├─ create_rag_chain()          │
   │ │  Retriever + Prompt + LLM    │
   └────────┬─────────────────────────┘
            │
    ┌───────┴──────────┐
    │                  │
    ▼                  ▼
Retrieval         Prompt Generation
    │                  │
    │ retriever.py     │ rag_chain.py
    │ └─ create_...()  │ └─ RAG_PROMPT_TEMPLATE
    │                  │
    ▼                  │
Base Retrieval         │
(Pinecone k=10)        │
    │                  │
    ▼                  │
FlashRank Rerank       │
(top_n=5)              │
    │                  │
    └────────┬─────────┘
             │
     Top-5 Documents
     Formatted w/ [1],[2]
             │
             ▼
   ┌──────────────────────────┐
   │  llm.py                  │
   │  ├─ Gemini-1.5-flash    │
   │  │  Temp: 0.3            │
   │  └─ Generate Answer      │
   └──────────┬───────────────┘
              │
              ▼
        Answer with
        Citations
        [1], [2], ...
              │
              ▼
   ┌──────────────────────────┐
   │  app.py                  │
   │  ├─ Display Answer       │
   │  ├─ Show Source Snippets │
   │  └─ Show Response Time   │
   └──────────────────────────┘
```

## Module Responsibilities Matrix

```
Module               | Responsibility          | External Deps
─────────────────────┼─────────────────────────┼──────────────────
config.py            | Configuration, Secrets  | Streamlit, os
embeddings.py        | Gemini Embeddings       | LangChain, Google
llm.py               | Gemini LLM              | LangChain, Google
vectorstore.py       | Pinecone Operations     | LangChain, Pinecone
document_processor.py| Doc Load & Chunk       | LangChain, PyPDF
retriever.py         | Retrieval Chain         | LangChain, FlashRank
rag_chain.py         | RAG Pipeline            | LangChain, Prompt
app.py               | Streamlit UI            | All modules
```

## Instantiation Order

```
On App Startup:
1. app.py imports all modules
2. config.py loaded (no dependencies)
3. embeddings.py, llm.py, vectorstore.py loaded (depend on config)
4. document_processor.py loaded (independent)
5. retriever.py loaded (depends on config)
6. rag_chain.py loaded (independent)

On get_components() call (@st.cache_resource):
1. get_embeddings() → GoogleGenerativeAIEmbeddings (created once)
2. get_llm() → ChatGoogleGenerativeAI (created once)
3. get_vectorstore() → PineconeVectorStore (created once)

On Document Upload:
1. process_documents(file/text)
2. add_documents_to_vectorstore(chunks)

On Query:
1. create_retriever(vectorstore)
2. create_rag_chain(retriever, llm)
3. query_rag(chain, retriever, question)
```

## Key Integration Points

```
config.py ←→ embeddings.py
  │              │
  │              └─ Config.EMBEDDING_MODEL
  │              └─ Config.GOOGLE_API_KEY
  │
  ├─→ llm.py
  │  └─ Config.LLM_MODEL
  │  └─ Config.GOOGLE_API_KEY
  │
  ├─→ vectorstore.py
  │  └─ Config.PINECONE_API_KEY
  │  └─ Config.PINECONE_INDEX_NAME
  │
  ├─→ document_processor.py
  │  └─ Config.CHUNK_SIZE
  │  └─ Config.CHUNK_OVERLAP
  │
  ├─→ retriever.py
  │  └─ Config.RETRIEVAL_TOP_K
  │  └─ Config.RERANK_TOP_N
  │
  └─→ app.py
     └─ Config.validate()
```

## Error Handling Flow

```
app.py (Streamlit UI)
  │
  └─ try/except blocks
     │
     ├─ Config.validate()
     │  └─ Show error if keys missing
     │
     ├─ document_processor.process_documents()
     │  └─ ValueError if no input
     │
     ├─ vectorstore.add_documents_to_vectorstore()
     │  └─ Pinecone API errors
     │
     ├─ rag_chain.create_rag_chain()
     │  └─ LLM generation errors
     │
     └─ retriever.create_retriever()
        └─ Vectorstore connection errors
  
  All errors caught and displayed to user with context
```

## Testing Strategy

```
Test Pyramid:

         ╱╲
        ╱  ╲         Integration Tests
       ╱────╲        (app.py + all modules)
      ╱      ╲
     ╱────────╲  Unit Tests
    ╱          ╲ (each module)
   ╱____________╲

Unit Tests (Easiest):
  - test_config.py
  - test_embeddings.py (mock API)
  - test_llm.py (mock API)
  - test_document_processor.py
  - test_vectorstore.py (mock Pinecone)
  - test_retriever.py (mock everything)
  - test_rag_chain.py (mock LLM + retriever)

Integration Tests (Harder):
  - test_document_upload.py (real Pinecone)
  - test_query_execution.py (real Gemini + Pinecone)

E2E Tests (Hardest):
  - test_streamlit_ui.py (browser automation)
```

---

This modular architecture enables:
- **Parallel Development**: Teams can work on different modules
- **Isolated Testing**: Each module tested separately
- **Easy Debugging**: Find issues in specific module
- **Flexible Deployment**: Deploy modules independently if needed
- **Clear Documentation**: Responsibilities well-defined
"""
