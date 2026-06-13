"""
Observability module for miniRAG.

Provides:
- Structured JSON logging via structlog
- Per-query latency tracking (embed / retrieve / rerank / generate)
- Optional LangSmith tracing (controlled by settings.enable_langsmith_tracing)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """Set up structlog with JSON output and stdlib integration."""
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "minirag") -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger."""
    return structlog.get_logger(name)


@dataclass
class LatencyBreakdown:
    """Tracks per-stage latency for a single RAG query."""

    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    generate_ms: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return self.embed_ms + self.retrieve_ms + self.rerank_ms + self.generate_ms

    def as_dict(self) -> dict:
        return {
            "embed_ms": round(self.embed_ms, 1),
            "retrieve_ms": round(self.retrieve_ms, 1),
            "rerank_ms": round(self.rerank_ms, 1),
            "generate_ms": round(self.generate_ms, 1),
            "total_ms": round(self.total_ms, 1),
            **self.extra,
        }


class QueryTracer:
    """
    Context-manager–based tracer for a single RAG query.

    Usage::

        tracer = QueryTracer(query="What is attention?")
        with tracer.stage("embed"):
            embeddings = embed_query(query)
        with tracer.stage("retrieve"):
            docs = retriever.invoke(query)
        tracer.log(num_chunks=len(docs), model="gemini-2.5-flash")
        return tracer.latency
    """

    def __init__(self, query: str, logger: structlog.stdlib.BoundLogger | None = None):
        self.query = query
        self.latency = LatencyBreakdown()
        self._log = logger or get_logger()
        self._stage_map = {
            "embed": "embed_ms",
            "retrieve": "retrieve_ms",
            "rerank": "rerank_ms",
            "generate": "generate_ms",
        }

    @contextmanager
    def stage(self, name: str) -> Generator[None, None, None]:
        """Time a named stage and store it in the latency breakdown."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - t0) * 1000
            attr = self._stage_map.get(name, name + "_ms")
            if hasattr(self.latency, attr):
                setattr(self.latency, attr, elapsed)
            else:
                self.latency.extra[attr] = elapsed

    def log(self, **extra_fields: object) -> None:
        """Emit a structured log line with latency + extra fields."""
        self._log.info(
            "rag_query",
            query_preview=self.query[:120],
            **self.latency.as_dict(),
            **extra_fields,
        )
