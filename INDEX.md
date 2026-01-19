# Mini RAG - Modular Architecture Index 📑

## Quick Navigation

### 🚀 Getting Started
1. **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)** ← Start here!
   - Overview of changes
   - Before/after comparison
   - Key improvements

2. **[MODULAR_README.md](MODULAR_README.md)**
   - Quick start guide
   - Module overview
   - Configuration

3. **[README.md](README.md)**
   - Original comprehensive guide
   - Setup instructions
   - Cost & performance details

### 🏗️ Architecture Documentation

4. **[ARCHITECTURE.md](ARCHITECTURE.md)**
   - Detailed module breakdown
   - Data flow (ingestion & retrieval)
   - Design benefits explained

5. **[ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md)**
   - Dependency graphs
   - Data flow diagrams
   - Integration points
   - Testing strategy

6. **[MODULAR_REFACTORING.md](MODULAR_REFACTORING.md)**
   - Design patterns used
   - Singleton pattern
   - Dependency injection
   - Migration guide
   - Testing benefits

### 💻 Core Modules

#### Configuration & Initialization
7. **[config.py](config.py)** (60 lines)
   - Centralized configuration class
   - API key management
   - Constants (chunk size, retrieval params, etc.)
   - `Config.validate()` method

8. **[embeddings.py](embeddings.py)** (25 lines)
   - Google Gemini embeddings initialization
   - Singleton: `get_embeddings()`

9. **[llm.py](llm.py)** (25 lines)
   - Gemini-1.5-flash LLM initialization
   - Singleton: `get_llm()`

10. **[vectorstore.py](vectorstore.py)** (60 lines)
    - Pinecone vector store initialization
    - `add_documents_to_vectorstore()` with metadata
    - `clear_vectorstore()` cleanup
    - Singleton: `get_vectorstore()`

#### Processing & Retrieval
11. **[document_processor.py](document_processor.py)** (80 lines)
    - `load_document_from_file()` - PDF/TXT handling
    - `load_text_from_string()` - pasted text
    - `chunk_documents()` - recursive splitting
    - `process_documents()` - complete pipeline

12. **[retriever.py](retriever.py)** (40 lines)
    - FlashRank reranker initialization
    - `create_retriever()` - builds compression retriever
    - Singleton: `get_reranker()`

#### RAG Pipeline
13. **[rag_chain.py](rag_chain.py)** (60 lines)
    - RAG prompt template
    - `format_docs()` - numbered source formatting
    - `create_rag_chain()` - runnable chain
    - `query_rag()` - execution & result retrieval

### 🎨 UI Layer
14. **[app.py](app.py)** (100 lines)
    - Streamlit interface
    - Sidebar document ingestion
    - Main query interface
    - Session state management
    - Error handling & validation
    - Component caching

### 📋 Configuration Files
15. **[requirements.txt](requirements.txt)**
    - All dependencies
    - `pip install -r requirements.txt`

16. **[.streamlit/secrets.toml](..streamlit/secrets.toml)**
    - Secrets template for hosting
    - Fill in your API keys

17. **[.env](..env)** (local testing)
    - Local environment variables
    - `GOOGLE_API_KEY`, `PINECONE_API_KEY`, etc.

---

## File Dependencies Map

```
config.py (no dependencies)
    ↓
embeddings.py ←─┐
llm.py          ├─→ vectorstore.py
retriever.py ←─┘
    ↓
rag_chain.py
document_processor.py
    ↓
app.py (Streamlit UI)
```

## Module Responsibilities

| Module | Responsibility | Key Functions |
|--------|------------------|---------------|
| config | Global config | `Config`, `Config.validate()` |
| embeddings | Gemini embeddings | `get_embeddings()` |
| llm | Gemini LLM | `get_llm()` |
| vectorstore | Pinecone operations | `get_vectorstore()`, `add_documents_to_vectorstore()` |
| document_processor | Load & chunk | `process_documents()` |
| retriever | Retrieval chain | `create_retriever()` |
| rag_chain | RAG pipeline | `create_rag_chain()`, `query_rag()` |
| app | Streamlit UI | Sidebar + main interface |

## Data Flow

### Ingestion (Documents → Vectors)
```
User Input (file/text)
  ↓ [app.py]
process_documents() [document_processor.py]
  ↓ load + chunk
add_documents_to_vectorstore() [vectorstore.py]
  ↓ embed + upsert
Pinecone ✓
```

### Retrieval (Query → Answer)
```
User Query
  ↓ [app.py]
create_retriever() [retriever.py]
create_rag_chain() [rag_chain.py]
  ↓ embed + retrieve + rerank + generate
query_rag() [rag_chain.py]
  ↓ execute
Answer + Sources ✓
```

## How to Use Each Module

### Import & Initialize
```python
from config import Config
from embeddings import get_embeddings
from llm import get_llm

embeddings = get_embeddings()  # Lazy singleton
llm = get_llm()                # Lazy singleton
```

### Process Documents
```python
from document_processor import process_documents
from vectorstore import add_documents_to_vectorstore

chunks = process_documents(uploaded_file=file)  # or input_text=text
add_documents_to_vectorstore(chunks)
```

### Build & Execute RAG
```python
from vectorstore import get_vectorstore
from retriever import create_retriever
from rag_chain import create_rag_chain, query_rag

retriever = create_retriever(get_vectorstore())
chain = create_rag_chain(retriever, get_llm())
answer, sources = query_rag(chain, retriever, "question")
```

## Testing Each Module

```bash
# Unit test individual modules
pytest tests/test_config.py
pytest tests/test_embeddings.py
pytest tests/test_document_processor.py
pytest tests/test_vectorstore.py

# Integration test end-to-end
pytest tests/test_rag_end_to_end.py

# Run all tests
pytest
```

## Running the App

```bash
# 1. Activate virtual environment
.\venv\Scripts\activate

# 2. Set API keys (in .env or environment)
set GOOGLE_API_KEY=your_key
set PINECONE_API_KEY=your_key
set PINECONE_INDEX_NAME=mini-rag

# 3. Run Streamlit
streamlit run app.py
```

Opens at: **http://localhost:8501**

---

## Documentation Reading Order

**For Quick Start:**
1. REFACTORING_SUMMARY.md (5 min)
2. MODULAR_README.md (10 min)
3. Run `streamlit run app.py` (2 min)

**For Deep Understanding:**
1. ARCHITECTURE.md (15 min)
2. ARCHITECTURE_DIAGRAMS.md (15 min)
3. MODULAR_REFACTORING.md (20 min)
4. Read individual module docstrings (15 min)

**For Development:**
1. Review MODULAR_REFACTORING.md design patterns
2. Check ARCHITECTURE.md data flows
3. Look at specific module files needed
4. Follow testing examples in MODULAR_REFACTORING.md

---

## Key Files at a Glance

```
📄 Documentation (Read First)
  ├─ REFACTORING_SUMMARY.md ⭐ Start here
  ├─ MODULAR_README.md
  ├─ ARCHITECTURE.md
  ├─ ARCHITECTURE_DIAGRAMS.md
  └─ MODULAR_REFACTORING.md

💻 Core Modules (Use in Code)
  ├─ config.py
  ├─ embeddings.py
  ├─ llm.py
  ├─ vectorstore.py
  ├─ document_processor.py
  ├─ retriever.py
  ├─ rag_chain.py
  └─ app.py

⚙️ Configuration
  ├─ requirements.txt
  ├─ .streamlit/secrets.toml
  └─ .env

🚀 To Run
  $ streamlit run app.py
```

---

**Created:** January 19, 2026  
**Status:** ✅ Complete  
**Architecture:** Modular, Production-Ready, Testable

For questions or improvements, check the relevant module's docstring! 🎯
