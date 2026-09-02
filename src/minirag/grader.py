"""
Document Relevance Grader for Corrective RAG (CRAG).

Evaluates retrieved context chunks against the user query to filter out
irrelevant noise and calculate context confidence for downstream routing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class DocumentGrade(BaseModel):
    """Structured grade response from the LLM grader."""

    is_relevant: bool = Field(
        description="Whether the document contains information directly relevant or helpful to answering the question."
    )
    score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Relevance confidence score from 0.0 (completely irrelevant) to 1.0 (highly relevant).",
    )
    reason: str = Field(
        default="",
        description="Brief 1-sentence explanation of why the document is or is not relevant.",
    )


@dataclass
class GradedDocument:
    """Document paired with its evaluation grade."""

    document: Document
    is_relevant: bool
    score: float = 1.0
    reason: str = ""


@dataclass
class GradingResult:
    """Aggregated result of grading a batch of candidate documents."""

    graded_docs: list[GradedDocument] = field(default_factory=list)
    relevant_docs: list[Document] = field(default_factory=list)
    relevance_ratio: float = 0.0
    confidence_level: str = "LOW"  # "HIGH", "PARTIAL", or "LOW"


DOC_GRADER_PROMPT = """\
You are an expert retrieval relevance grader.
Evaluate whether the following retrieved document excerpt contains facts, context, or keywords relevant to answering the user question.

Question:
{question}

Document Excerpt:
{document}

Respond ONLY with a valid JSON object matching this exact schema:
{{
  "is_relevant": true or false,
  "score": float between 0.0 and 1.0,
  "reason": "short explanation"
}}
"""


class DocumentGrader:
    """
    Evaluates retrieved chunks to filter out noise and assess context sufficiency.
    """

    def __init__(self, llm: Any) -> None:
        """
        Initialize DocumentGrader with an LLM instance.

        Args:
            llm: Language model supporting .invoke() (e.g. ChatGoogleGenerativeAI).
        """
        self._llm = llm
        self._prompt = ChatPromptTemplate.from_template(DOC_GRADER_PROMPT)
        self._chain = self._prompt | self._llm | StrOutputParser()

    def grade_document(self, doc: Document, question: str) -> GradedDocument:
        """
        Grade a single Document for relevance to the question.

        Args:
            doc: Document chunk to evaluate.
            question: User query string.

        Returns:
            GradedDocument with is_relevant, score, and reason.
        """
        try:
            raw_response = self._chain.invoke(
                {"question": question, "document": doc.page_content[:1500]}
            ).strip()

            # Clean JSON formatting if enclosed in code fences
            if raw_response.startswith("```"):
                lines = raw_response.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_response = "\n".join(lines).strip()

            data = json.loads(raw_response)
            grade = DocumentGrade(**data)
            return GradedDocument(
                document=doc,
                is_relevant=bool(grade.is_relevant),
                score=float(grade.score),
                reason=str(grade.reason),
            )
        except Exception as exc:
            # Fallback gracefully: treat doc as relevant to avoid losing valid information
            return GradedDocument(
                document=doc,
                is_relevant=True,
                score=0.7,
                reason=f"Defaulted due to evaluation error: {exc}",
            )

    def grade_documents(
        self,
        docs: list[Document],
        question: str,
        *,
        high_threshold: float = 0.6,
        partial_threshold: float = 0.3,
    ) -> GradingResult:
        """
        Grade a list of retrieved documents and calculate aggregate confidence.

        Args:
            docs: List of retrieved candidate documents.
            question: User query string.
            high_threshold: Ratio for HIGH confidence.
            partial_threshold: Ratio for PARTIAL confidence.

        Returns:
            GradingResult with graded list, filtered relevant list, and confidence level.
        """
        if not docs:
            return GradingResult(
                graded_docs=[],
                relevant_docs=[],
                relevance_ratio=0.0,
                confidence_level="LOW",
            )

        graded_list: list[GradedDocument] = []
        for doc in docs:
            graded_list.append(self.grade_document(doc, question))

        relevant_docs = [g.document for g in graded_list if g.is_relevant]
        ratio = len(relevant_docs) / len(docs) if docs else 0.0

        if ratio >= high_threshold:
            confidence = "HIGH"
        elif ratio >= partial_threshold:
            confidence = "PARTIAL"
        else:
            confidence = "LOW"

        return GradingResult(
            graded_docs=graded_list,
            relevant_docs=relevant_docs,
            relevance_ratio=ratio,
            confidence_level=confidence,
        )
