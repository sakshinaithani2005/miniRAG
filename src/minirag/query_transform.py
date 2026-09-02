"""
Query Transformation & Decomposition module for miniRAG.

Supports:
- Web-query reformulation (transforming ambiguous or internal questions into web-search queries)
- Sub-question decomposition (breaking complex or multi-hop questions into focused sub-queries)
"""

from __future__ import annotations

import json
from typing import Any

from .observability import get_logger
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logger = get_logger(__name__)

WEB_QUERY_PROMPT = """\
You are an expert search query generator.
The user asked a question, but initial document retrieval failed or returned incomplete context.
Generate an optimized, concise web search query targeting reputable external sources to find the answer.

Original Question:
{question}

Failed/Missing Context Hints:
{context_hints}

Output ONLY the search query string, nothing else.
"""

DECOMPOSE_PROMPT = """\
You are an expert research analyst.
Break down the following complex or multi-part question into 2 to 3 distinct, simpler sub-questions \
that can be retrieved independently.

Question:
{question}

Respond ONLY with a valid JSON array of strings, for example:
["First sub-question?", "Second sub-question?"]
"""


class QueryTransformer:
    """Transforms and decomposes user queries for advanced retrieval workflows."""

    def __init__(self, llm: Any) -> None:
        """
        Initialize QueryTransformer with an LLM.

        Args:
            llm: Language model supporting .invoke().
        """
        self._llm = llm
        self._web_chain = (
            ChatPromptTemplate.from_template(WEB_QUERY_PROMPT)
            | self._llm
            | StrOutputParser()
        )
        self._decompose_chain = (
            ChatPromptTemplate.from_template(DECOMPOSE_PROMPT)
            | self._llm
            | StrOutputParser()
        )

    def transform_for_web(self, question: str, context_hints: str = "None") -> str:
        """
        Reformulate user query into an effective web search query.

        Args:
            question: Original user question.
            context_hints: Brief summary of what was missing in local retrieval.

        Returns:
            Optimized web search query string.
        """
        try:
            query = self._web_chain.invoke(
                {"question": question, "context_hints": context_hints}
            ).strip()
            # Strip outer quotes if returned
            if (query.startswith('"') and query.endswith('"')) or (
                query.startswith("'") and query.endswith("'")
            ):
                query = query[1:-1].strip()
            return query if query else question
        except Exception as exc:
            logger.warning("Web query transform failed, using original query", error=str(exc))
            return question

    def decompose(self, question: str) -> list[str]:
        """
        Decompose a question into sub-queries.

        Args:
            question: User question.

        Returns:
            List of sub-questions (returns [question] if decomposition is not needed or fails).
        """
        try:
            raw = self._decompose_chain.invoke({"question": question}).strip()
            if raw.startswith("```"):
                lines = raw.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw = "\n".join(lines).strip()

            items = json.loads(raw)
            if isinstance(items, list) and all(isinstance(x, str) for x in items) and items:
                return items
            return [question]
        except Exception:
            return [question]
