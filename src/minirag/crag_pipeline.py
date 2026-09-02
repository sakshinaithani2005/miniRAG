"""
Corrective RAG (CRAG) Pipeline module for miniRAG.

Coordinates the end-to-end self-reflective workflow:
  1. Retrieve candidate chunks (Dense/Hybrid/MMR) + FlashRank Cross-Encoder.
  2. Grade chunks for relevance using DocumentGrader.
  3. Action routing:
     - HIGH confidence (>= threshold): Filter out noise and generate answer.
     - PARTIAL confidence: Keep relevant docs, reformulate query, and augment with web search.
     - LOW confidence (< threshold): Rewrite query and execute full web search fallback.
  4. Generate answer with inline citations.
  5. Grounding & Hallucination verification via GroundingChecker.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

from .grader import DocumentGrader, GradingResult
from .grounding import GroundingChecker, GroundingReport
from .observability import QueryTracer
from .query_transform import QueryTransformer
from .rag_chain import RAG_PROMPT_TEMPLATE, format_docs, verify_citations
from .web_search import web_search
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


@dataclass
class CRAGOutput:
    """Full execution state and reflection metrics for a CRAG query."""

    answer: str
    final_docs: list[Document]
    grading_result: GradingResult
    action_taken: str  # "CORRECT_DOCS_DIRECT", "AMBIGUOUS_WEB_AUGMENTED", "INSUFFICIENT_WEB_FALLBACK"
    transformed_query: str | None = None
    web_fallback_used: bool = False
    grounding_report: GroundingReport | None = None
    warnings: list[str] = field(default_factory=list)


class CRAGPipeline:
    """
    Orchestrates the Corrective RAG (CRAG) and Self-Reflection workflow.
    """

    def __init__(
        self,
        retriever: Any,
        llm: Any,
        *,
        relevance_threshold: float = 0.5,
        enable_grounding_check: bool = True,
    ) -> None:
        """
        Initialize the CRAG pipeline.

        Args:
            retriever: Base retriever (hybrid, dense, or MMR).
            llm: LLM instance for generation and evaluation.
            relevance_threshold: Ratio threshold (0.0 - 1.0) of relevant docs needed.
            enable_grounding_check: Whether to run post-generation hallucination critique.
        """
        self.retriever = retriever
        self.llm = llm
        self.relevance_threshold = relevance_threshold
        self.enable_grounding_check = enable_grounding_check

        self.grader = DocumentGrader(llm)
        self.query_transformer = QueryTransformer(llm)
        self.grounding_checker = GroundingChecker(llm)

    def prepare_context(
        self,
        question: str,
        *,
        tracer: QueryTracer | None = None,
    ) -> tuple[list[Document], GradingResult, str, str | None, bool]:
        """
        Execute the retrieval, grading, and routing stages before generation.

        Returns:
            Tuple of (final_docs, grading_result, action_taken, transformed_query, web_fallback_used).
        """
        # Step 1: Initial Retrieval
        if tracer:
            with tracer.stage("retrieve"):
                initial_docs: list[Document] = self.retriever.invoke(question)
        else:
            initial_docs = self.retriever.invoke(question)

        # Step 2: Document Relevance Grading
        if tracer:
            with tracer.stage("grade"):
                grading_result = self.grader.grade_documents(
                    initial_docs,
                    question,
                    high_threshold=self.relevance_threshold,
                    partial_threshold=0.25,
                )
        else:
            grading_result = self.grader.grade_documents(
                initial_docs,
                question,
                high_threshold=self.relevance_threshold,
                partial_threshold=0.25,
            )

        transformed_query: str | None = None
        web_fallback_used = False
        final_docs: list[Document] = []

        # Step 3: Routing Logic based on confidence
        if grading_result.confidence_level == "HIGH":
            # Document context is strong: use only the verified relevant docs
            action_taken = "CORRECT_DOCS_DIRECT"
            final_docs = grading_result.relevant_docs

        elif grading_result.confidence_level == "PARTIAL":
            # Context has some relevant pieces but is incomplete: augment with web search
            action_taken = "AMBIGUOUS_WEB_AUGMENTED"
            transformed_query = self.query_transformer.transform_for_web(
                question,
                context_hints="Partial document matches found, searching for missing details.",
            )
            web_docs = web_search(transformed_query, max_results=2)
            final_docs = grading_result.relevant_docs + web_docs
            web_fallback_used = len(web_docs) > 0

        else:
            # Context is insufficient or irrelevant: transform query & trigger full web fallback
            action_taken = "INSUFFICIENT_WEB_FALLBACK"
            transformed_query = self.query_transformer.transform_for_web(
                question,
                context_hints="No relevant chunks found in internal documents.",
            )
            web_docs = web_search(transformed_query, max_results=3)
            final_docs = web_docs
            web_fallback_used = len(web_docs) > 0

        return final_docs, grading_result, action_taken, transformed_query, web_fallback_used

    def query(
        self,
        question: str,
        *,
        tracer: QueryTracer | None = None,
    ) -> CRAGOutput:
        """
        Execute full CRAG pipeline synchronously.

        Args:
            question: User question.
            tracer: Latency and metrics tracer.

        Returns:
            CRAGOutput with answer, metrics, and reflection audit data.
        """
        (
            final_docs,
            grading_result,
            action_taken,
            transformed_query,
            web_fallback_used,
        ) = self.prepare_context(question, tracer=tracer)

        if not final_docs:
            answer = (
                "No relevant information found in the indexed documents or external sources."
            )
            return CRAGOutput(
                answer=answer,
                final_docs=[],
                grading_result=grading_result,
                action_taken=action_taken,
                transformed_query=transformed_query,
                web_fallback_used=web_fallback_used,
                warnings=[],
            )

        # Generation
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        gen_chain = prompt | self.llm | StrOutputParser()
        context_str = format_docs(final_docs)

        if tracer:
            with tracer.stage("generate"):
                answer = gen_chain.invoke({"context": context_str, "question": question})
        else:
            answer = gen_chain.invoke({"context": context_str, "question": question})

        # Grounding & Citations check
        warnings = verify_citations(answer, len(final_docs))
        grounding_report: GroundingReport | None = None

        if self.enable_grounding_check:
            grounding_report = self.grounding_checker.check(answer, final_docs, question)
            if not grounding_report.is_grounded:
                warnings.append(
                    f"Grounding Warning: Faithfulness score {grounding_report.faithfulness_score:.0%}. "
                    f"Possible unsupported claims: {', '.join(grounding_report.hallucinated_claims)}"
                )

        if tracer:
            tracer.log(
                num_docs=len(final_docs),
                crag_action=action_taken,
                relevance_ratio=grading_result.relevance_ratio,
                grounded=grounding_report.is_grounded if grounding_report else True,
            )

        return CRAGOutput(
            answer=answer,
            final_docs=final_docs,
            grading_result=grading_result,
            action_taken=action_taken,
            transformed_query=transformed_query,
            web_fallback_used=web_fallback_used,
            grounding_report=grounding_report,
            warnings=warnings,
        )

    def stream_answer(
        self,
        question: str,
        final_docs: list[Document],
    ) -> Generator:
        """
        Stream the final generation step tokens.

        Args:
            question: Original question.
            final_docs: Prepared and filtered context documents.

        Returns:
            Token generator.
        """
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        gen_chain = prompt | self.llm | StrOutputParser()
        context_str = format_docs(final_docs)
        return gen_chain.stream({"context": context_str, "question": question})
