"""
CLI interface for miniRAG.

Allows running RAG queries entirely from the terminal — no UI required.

Usage examples::

    # Query against a PDF
    python -m minirag.cli --query "What is multi-head attention?" --doc attention.pdf

    # Query against pasted text
    echo "Your text here" | python -m minirag.cli --query "Summarise this"

    # Use a specific retrieval strategy
    python -m minirag.cli --query "What is BERT?" --doc paper.pdf --strategy hybrid
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m minirag.cli",
        description="miniRAG — command-line RAG query interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-q", "--query", required=True, help="Question to answer")
    parser.add_argument(
        "-d", "--doc",
        default=None,
        help="Path to a PDF or TXT file to index before querying",
    )
    parser.add_argument(
        "--strategy",
        choices=["dense", "hybrid", "mmr"],
        default="hybrid",
        help="Retrieval strategy (default: hybrid)",
    )
    parser.add_argument(
        "--no-rewrite",
        action="store_true",
        help="Disable HyDE-lite query rewriting",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override retrieval top-k (default from config)",
    )
    return parser.parse_args()


def _load_env() -> None:
    """Load .env from CWD or script directory."""
    from dotenv import load_dotenv

    for candidate in [Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"]:
        if candidate.exists():
            load_dotenv(candidate, override=True)
            return


def main() -> int:
    _load_env()

    # Lazy imports after env is loaded
    from config import Config, RetrievalStrategy
    from document_processor import process_documents
    from embeddings import get_embeddings
    from llm import get_llm
    from observability import QueryTracer, configure_logging
    from rag_chain import create_rag_chain, query_rag
    from retriever import create_retriever
    from vectorstore import add_documents_to_vectorstore, get_vectorstore

    configure_logging()
    args = _parse_args()

    if not Config.validate():
        print("❌ Missing API keys. Set GOOGLE_API_KEY and PINECONE_API_KEY in .env", file=sys.stderr)
        return 1

    # ── Index document ────────────────────────────────────────────────────
    chunks = []
    if args.doc:
        doc_path = Path(args.doc)
        if not doc_path.exists():
            print(f"❌ File not found: {doc_path}", file=sys.stderr)
            return 1

        print(f"📄 Indexing {doc_path.name} …")
        t0 = time.perf_counter()

        # Simulate an UploadedFile-like object for the document processor
        class _FileWrapper:
            def __init__(self, p: Path):
                self._bytes = p.read_bytes()
                self.name = p.name
                self.type = "application/pdf" if p.suffix.lower() == ".pdf" else "text/plain"

            def getvalue(self) -> bytes:
                return self._bytes

        chunks = process_documents(uploaded_file=_FileWrapper(doc_path))
        vs = get_vectorstore()
        add_documents_to_vectorstore(chunks, vs)
        elapsed = time.perf_counter() - t0
        print(f"✅ Indexed {len(chunks)} chunks in {elapsed:.2f}s")

    # ── Retrieve & generate ───────────────────────────────────────────────
    strategy = RetrievalStrategy(args.strategy)
    vs = get_vectorstore()
    retriever = create_retriever(vs, corpus=chunks or None, strategy=strategy)
    llm = get_llm()
    chain = create_rag_chain(retriever, llm)

    tracer = QueryTracer(args.query)
    answer, docs, warnings = query_rag(
        chain,
        retriever,
        args.query,
        enable_rewrite=not args.no_rewrite,
        llm=llm,
        tracer=tracer,
    )

    # ── Output ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)

    print("\n" + "-" * 70)
    print(f"SOURCES ({len(docs)} chunks retrieved)")
    print("-" * 70)
    for i, doc in enumerate(docs, 1):
        print(f"[{i}] {doc.metadata.get('source', 'Unknown')} "
              f"(chunk {doc.metadata.get('chunk_id', '?')})")
        print(f"    {doc.page_content[:120].strip()}…")

    latency = tracer.latency
    print(f"\n⏱  Total: {latency.total_ms:.0f}ms  "
          f"(retrieve: {latency.retrieve_ms:.0f}ms, "
          f"generate: {latency.generate_ms:.0f}ms)")

    if warnings:
        for w in warnings:
            print(f"\n{w}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
