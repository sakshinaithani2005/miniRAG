"""
RAG chain module for miniRAG.

Builds the full retrieval-augmented generation pipeline:
  1. (Optional) Query rewriting / HyDE-lite
  2. Retrieve + rerank
  3. Format context with numbered citations
  4. Generate answer
  5. (Optional) Answer grounding verification

All steps are instrumented via the QueryTracer for latency breakdown.
"""

from __future__ import annotations

import re
from typing import Generator, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from config import Config
from observability import QueryTracer

# ChatGoogleGenerativeAI imported lazily to avoid hard dep in tests
try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
except ImportError:
    ChatGoogleGenerativeAI = None  # type: ignore


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RAG_PROMPT_TEMPLATE = """\
You are a precise, citation-aware assistant. Answer **only** from the provided context.

Rules:
- Cite sources inline as [1], [2], etc. at the end of every relevant sentence.
- If multiple contexts support a claim, cite all of them: [1][3].
- If the answer is not found in the context, say exactly: "No relevant information found in the provided documents."
- Do NOT hallucinate or use outside knowledge.

Context:
{context}

Question: {question}

Answer:\
"""

QUERY_REWRITE_TEMPLATE = """\
You are a search query optimizer.
Rewrite the following question into a cleaner, more specific search query that will \
retrieve the most relevant document chunks. Output ONLY the rewritten query, no explanation.

Original question: {question}
Rewritten query:\
"""


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_docs(documents: List[Document]) -> str:
    """Format documents as numbered, source-labelled context blocks."""
    return "\n\n".join(
        f"[{i + 1}] {doc.page_content}\n"
        f"    Source: {doc.metadata.get('source', 'Unknown')} "
        f"(chunk {doc.metadata.get('chunk_id', '?')})"
        for i, doc in enumerate(documents)
    )


# ---------------------------------------------------------------------------
# Query rewriting (HyDE-lite)
# ---------------------------------------------------------------------------

def rewrite_query(question: str, llm) -> str:
    """
    Rewrite a user question into an optimised retrieval query.

    Uses the LLM to expand / clarify ambiguous phrasing before hitting
    the vector store. Falls back to the original question on any error.

    Args:
        question: Raw user question.
        llm:      LLM instance with ``.invoke()`` support.

    Returns:
        Rewritten query string (falls back to original on error).
    """
    try:
        prompt = ChatPromptTemplate.from_template(QUERY_REWRITE_TEMPLATE)
        chain = prompt | llm | StrOutputParser()
        rewritten = chain.invoke({"question": question}).strip()
        return rewritten if rewritten else question
    except Exception as exc:
        print(f"⚠️  Query rewrite failed ({exc}) — using original.")
        return question


# ---------------------------------------------------------------------------
# Answer grounding check
# ---------------------------------------------------------------------------

def verify_citations(answer: str, num_docs: int) -> List[str]:
    """
    Parse citation markers from the answer and flag invalid ones.

    Args:
        answer:   Generated answer text.
        num_docs: Number of context documents provided.

    Returns:
        List of warning strings (empty = all citations valid).
    """
    cited = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    valid = set(range(1, num_docs + 1))
    hallucinated = cited - valid
    if hallucinated:
        return [
            f"⚠️ Hallucinated citation(s): {sorted(hallucinated)} "
            f"(only {num_docs} context docs provided)"
        ]
    return []


# ---------------------------------------------------------------------------
# Chain construction
# ---------------------------------------------------------------------------

def create_rag_chain(retriever, llm):
    """
    Create an LCEL RAG chain.

    Args:
        retriever: Any retriever with an ``.invoke(query) -> List[Document]``
                   interface (dense, hybrid, or MMR).
        llm:       LLM instance.

    Returns:
        Runnable LCEL chain: question → answer string.
    """
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------

def query_rag(
    chain,
    retriever,
    question: str,
    *,
    enable_rewrite: bool = True,
    llm=None,
    tracer: Optional[QueryTracer] = None,
) -> Tuple[str, List[Document], List[str]]:
    """
    Execute a full RAG query with optional rewriting and citation checking.

    Args:
        chain:          LCEL chain from ``create_rag_chain()``.
        retriever:      Retriever instance.
        question:       Raw user question string.
        enable_rewrite: Whether to apply HyDE-lite query rewriting.
        llm:            LLM for rewriting (required when enable_rewrite=True).
        tracer:         ``QueryTracer`` for per-stage latency tracking.

    Returns:
        Tuple of (answer, retrieved_docs, warnings).
    """
    retrieval_query = question
    if enable_rewrite and llm is not None:
        retrieval_query = rewrite_query(question, llm)

    if tracer:
        with tracer.stage("retrieve"):
            retrieved_docs: List[Document] = retriever.invoke(retrieval_query)
    else:
        retrieved_docs = retriever.invoke(retrieval_query)

    if tracer:
        with tracer.stage("generate"):
            answer: str = chain.invoke(question)
    else:
        answer = chain.invoke(question)

    warnings = verify_citations(answer, len(retrieved_docs))

    if tracer:
        tracer.log(
            num_docs=len(retrieved_docs),
            query_rewritten=(retrieval_query != question),
            citation_warnings=len(warnings),
        )

    return answer, retrieved_docs, warnings


def stream_rag(
    chain,
    retriever,
    question: str,
    *,
    enable_rewrite: bool = True,
    llm=None,
) -> Tuple[Generator, List[Document]]:
    """
    Execute a RAG query with streaming token output.

    Returns:
        Tuple of (token_generator, retrieved_docs).
        Consume the generator in ``st.write_stream()`` or similar.
    """
    retrieval_query = question
    if enable_rewrite and llm is not None:
        retrieval_query = rewrite_query(question, llm)

    retrieved_docs: List[Document] = retriever.invoke(retrieval_query)
    token_stream = chain.stream(question)
    return token_stream, retrieved_docs
