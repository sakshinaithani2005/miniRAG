# Mini RAG - Modular Architecture

A production-ready, **modularly designed** retrieval-augmented generation (RAG) application combining Gemini embeddings, Pinecone vector storage, FlashRank reranking, and Streamlit frontend for grounded question-answering with citations.

## Project Structure

```
mini-rag/
├── config.py                 # Configuration & API key management
├── embeddings.py             # Gemini embeddings module
├── llm.py                    # Gemini LLM module
├── vectorstore.py            # Pinecone vector store module
├── document_processor.py      # Document loading & chunking
├── retriever.py              # Retrieval & reranking module
├── rag_chain.py              # RAG chain & prompt engineering
├── app.py                    # Streamlit UI (clean, modular)
├── ARCHITECTURE.md           # Detailed modular architecture docs
├── requirements.txt          # Dependencies
├── README.md                 # Original README
└── .streamlit/
    └── secrets.toml          # Secrets template
```

## Module Overview

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| **config.py** | Centralized configuration | `Config.validate()`, API keys, constants |
| **embeddings.py** | Gemini embeddings | `get_embeddings()` (singleton) |
| **llm.py** | Gemini-1.5-flash | `get_llm()` (singleton) |
| **vectorstore.py** | Pinecone ops | `get_vectorstore()`, `add_documents_to_vectorstore()` |
| **document_processor.py** | Load & chunk | `process_documents()`, `chunk_documents()` |
| **retriever.py** | Retrieval chain | `create_retriever()`, `get_reranker()` |
| **rag_chain.py** | RAG pipeline | `create_rag_chain()`, `query_rag()` |
| **app.py** | Streamlit UI | Sidebar ingestion, main query interface |

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed data flow and design benefits.

## Tech Stack

- **Embeddings**: Google Gemini (text-embedding-004, 768-dim, free tier)
- **LLM**: Gemini-1.5-flash (low latency, cost-efficient)
- **Vector Store**: Pinecone serverless (free tier, cosine metric)
- **Reranking**: FlashRank (zero-cost, CPU-based cross-encoder, ~ms latency)
- **Framework**: LangChain (orchestration, chunking, retrieval)
- **Frontend**: Streamlit (simple, interactive, free hosting)

## Quick Start

### 1. Create API Keys

**Google AI Studio:**
- Go to https://aistudio.google.com/
- Click "Get API key" → Create new API key

**Pinecone:**
- Sign up at https://www.pinecone.io/
- Create serverless index: `mini-rag` (768-dim, cosine)

### 2. Setup Environment

```bash
cd miniRAG
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Secrets

Create `.env` file in project root:
```
GOOGLE_API_KEY=your_actual_key_here
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX_NAME=mini-rag
```

### 4. Run the App

```bash
streamlit run app.py
```

App opens at **http://localhost:8501**

## How It Works

### Ingestion Pipeline
1. User pastes text or uploads PDF/TXT in sidebar
2. `process_documents()` loads and chunks (1000 chars, 200 overlap)
3. `add_documents_to_vectorstore()` embeds with Gemini and upserts to Pinecone
4. Chunks stored with metadata: `chunk_id`, `source_snippet`

### Query Pipeline
1. User enters question
2. `create_retriever()` builds retrieval chain (base + rerank)
3. `create_rag_chain()` builds prompt + LLM chain
4. `query_rag()` executes: retrieve top-10 → rerank to top-5 → generate answer
5. Display answer with inline citations [1], [2] + source snippets

## Configuration

All settings in `config.py`:

```python
CHUNK_SIZE = 1000              # ~800-1200 tokens
CHUNK_OVERLAP = 200            # Semantic continuity
RETRIEVAL_TOP_K = 10           # Initial retrieval count
RERANK_TOP_N = 5               # After FlashRank
LLM_TEMPERATURE = 0.3          # Low for grounded answers
```

## Features

✅ **Modular design** - each component in separate file  
✅ **Singleton pattern** - efficient resource use  
✅ **Text & PDF upload** - supports both input types  
✅ **Smart reranking** - FlashRank (~5ms, zero-cost)  
✅ **Inline citations** - [1], [2] in answers  
✅ **Source display** - chunk previews below answer  
✅ **Latency tracking** - response time per query  
✅ **Config validation** - early error detection  
✅ **Streamlit caching** - @st.cache_resource  

## Design Principles

### Modularity
- **Single Responsibility**: Each module has one purpose
- **Loose Coupling**: Modules depend on abstractions (config, functions)
- **High Cohesion**: Related functionality grouped together

### Reusability
- Modules can be imported in other projects
- Easy to swap implementations (e.g., different embeddings)

### Testability
- Unit test each module independently
- Mock components easily
- No circular dependencies

### Maintainability
- Clear file organization
- Descriptive docstrings
- Configuration centralized

## Cost & Performance

| Component | Cost | Latency |
|-----------|------|---------|
| Gemini Embeddings | ~$0.025/M tokens | ~200ms |
| Gemini-1.5-flash | ~$0.075/M in, $0.3/M out | ~1-2s |
| Pinecone serverless | Free: 1M ops/month | ~50ms |
| FlashRank reranking | $0 (local) | ~5ms |

**Typical query**: ~$0.0001-0.0005

## Hosting on Streamlit Cloud

1. Push to public GitHub repo
2. Go to https://share.streamlit.io/
3. Connect repo → Add secrets in Advanced settings:
   ```toml
   GOOGLE_API_KEY = "your_key"
   PINECONE_API_KEY = "your_key"
   PINECONE_INDEX_NAME = "mini-rag"
   ```
4. Deploy ✅

## Future Enhancements

- [ ] **Chat history** - conversation context
- [ ] **Multi-doc** - Pinecone namespaces
- [ ] **Web search** - real-time augmentation
- [ ] **Cost tracking** - token + $ logging
- [ ] **Streaming** - token-by-token generation
- [ ] **Auth** - user authentication
- [ ] **Async** - convert to async operations

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "API key not found" | Set env vars or edit `.env` |
| "Pinecone index not found" | Check index name (768-dim, cosine) |
| "FlashRank download stuck" | Check internet (~500MB download) |
| "Slow indexing" | Normal for large PDFs; chunking in progress |
| "Empty results" | Ensure vectors in Pinecone; try different query |

## License

MIT

---

**Built with ❤️ using modular design patterns for production RAG.**
