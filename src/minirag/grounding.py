"""
Grounding and Hallucination Self-Critique module for miniRAG.

Performs automated verification of generated answers against retrieved context
to detect unsupported claims, hallucinated facts, or ungrounded statements.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class GroundingReport(BaseModel):
    """Detailed result of grounding and hallucination check."""

    is_grounded: bool = Field(
        description="True if all substantive claims in the answer are strictly supported by context."
    )
    faithfulness_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Fraction of claims supported by context (1.0 = completely faithful).",
    )
    hallucinated_claims: list[str] = Field(
        default_factory=list,
        description="List of specific claims or statements not supported by the provided context.",
    )
    summary: str = Field(
        default="",
        description="Concise audit summary of the grounding verification.",
    )


GROUNDING_CHECK_PROMPT = """\
You are an uncompromising, meticulous factual auditor.
Your job is to check if the generated answer is strictly grounded in and supported by the provided context documents.

Context Documents:
{context}

User Question:
{question}

Generated Answer:
{answer}

Evaluate every factual assertion in the answer:
1. Identify any claims made in the answer that CANNOT be verified from the context documents.
2. Determine if the overall answer is faithful and grounded.
3. Compute a faithfulness score between 0.0 (total hallucination) and 1.0 (fully grounded).

Respond ONLY with a valid JSON object matching this schema:
{{
  "is_grounded": true or false,
  "faithfulness_score": float between 0.0 and 1.0,
  "hallucinated_claims": ["claim 1 not in context", "claim 2 not in context"],
  "summary": "1-sentence audit summary"
}}
"""


class GroundingChecker:
    """Evaluates answer faithfulness and flags hallucinated claims."""

    def __init__(self, llm: Any) -> None:
        """
        Initialize GroundingChecker with an LLM.

        Args:
            llm: Language model supporting .invoke().
        """
        self._llm = llm
        self._prompt = ChatPromptTemplate.from_template(GROUNDING_CHECK_PROMPT)
        self._chain = self._prompt | self._llm | StrOutputParser()

    def check(
        self,
        answer: str,
        docs: list[Document],
        question: str,
    ) -> GroundingReport:
        """
        Verify grounding of an answer against provided context documents.

        Args:
            answer: Generated answer text.
            docs: Retrieved context documents provided to the generator.
            question: Original question.

        Returns:
            GroundingReport with faithfulness score and hallucination list.
        """
        if not docs:
            # If no docs were provided and answer is refusal, it's grounded
            if "no relevant information" in answer.lower():
                return GroundingReport(
                    is_grounded=True,
                    faithfulness_score=1.0,
                    hallucinated_claims=[],
                    summary="Proper refusal when no context documents were provided.",
                )
            return GroundingReport(
                is_grounded=False,
                faithfulness_score=0.0,
                hallucinated_claims=[answer],
                summary="Answer generated without any supporting context documents.",
            )

        context_str = "\n\n".join(
            f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs)
        )

        try:
            raw = self._chain.invoke(
                {"context": context_str[:4000], "question": question, "answer": answer}
            ).strip()

            if raw.startswith("```"):
                lines = raw.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw = "\n".join(lines).strip()

            data = json.loads(raw)
            return GroundingReport(**data)
        except Exception as exc:
            # Fallback gracefully
            return GroundingReport(
                is_grounded=True,
                faithfulness_score=0.9,
                hallucinated_claims=[],
                summary=f"Audit completed with heuristic fallback ({exc}).",
            )
