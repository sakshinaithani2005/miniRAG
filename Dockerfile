# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

# Security: run as non-root
RUN addgroup --system app && adduser --system --group app
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencies layer (cached unless pyproject.toml changes) ──────────────────
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[web]"

# ── Application layer ─────────────────────────────────────────────────────────
COPY src/ ./src/
COPY app.py config.py embeddings.py llm.py vectorstore.py \
     retriever.py rag_chain.py document_processor.py ./

USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
