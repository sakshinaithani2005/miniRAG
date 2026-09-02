"""Tests for CRAGPipeline and Corrective RAG routing workflows."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from minirag.crag_pipeline import CRAGPipeline
from minirag.grader import GradedDocument, GradingResult
from minirag.grounding import GroundingReport


def test_crag_high_relevance_direct_routing():
    mock_retriever = MagicMock()
    mock_llm = FakeListChatModel(responses=["CRAG improves accuracy using self-reflection [1]."])

    doc1 = Document(page_content="CRAG improves retrieval accuracy.", metadata={"source": "paper.pdf"})
    doc2 = Document(page_content="Attention mechanisms process sequences.", metadata={"source": "transformer.pdf"})
    mock_retriever.invoke.return_value = [doc1, doc2]

    pipeline = CRAGPipeline(
        retriever=mock_retriever, llm=mock_llm,
        relevance_threshold=0.5, enable_grounding_check=False,
    )

    pipeline.grader.grade_documents = MagicMock(
        return_value=GradingResult(
            graded_docs=[GradedDocument(doc1, True, 0.9), GradedDocument(doc2, True, 0.8)],
            relevant_docs=[doc1, doc2],
            relevance_ratio=1.0,
            confidence_level="HIGH",
        )
    )

    output = pipeline.query("How does CRAG work?")

    assert output.action_taken == "CORRECT_DOCS_DIRECT"
    assert output.web_fallback_used is False
    assert len(output.final_docs) == 2
    assert "CRAG" in output.answer


def test_crag_low_relevance_web_fallback_routing():
    mock_retriever = MagicMock()
    mock_llm = FakeListChatModel(responses=["Deep learning uses neural networks [1]."])

    doc1 = Document(page_content="Unrelated recipe text.", metadata={"source": "cooking.txt"})
    mock_retriever.invoke.return_value = [doc1]

    pipeline = CRAGPipeline(
        retriever=mock_retriever, llm=mock_llm,
        relevance_threshold=0.5, enable_grounding_check=False,
    )

    pipeline.grader.grade_documents = MagicMock(
        return_value=GradingResult(
            graded_docs=[GradedDocument(doc1, False, 0.1)],
            relevant_docs=[],
            relevance_ratio=0.0,
            confidence_level="LOW",
        )
    )

    pipeline.query_transformer.transform_for_web = MagicMock(return_value="what is deep learning")

    web_doc = Document(
        page_content="Deep learning is a subset of machine learning.",
        metadata={"source": "web"},
    )
    with patch("minirag.crag_pipeline.web_search", return_value=[web_doc]):
        output = pipeline.query("What is deep learning?")

        assert output.action_taken == "INSUFFICIENT_WEB_FALLBACK"
        assert output.web_fallback_used is True
        assert len(output.final_docs) == 1
        assert output.final_docs[0].metadata["source"] == "web"


def test_crag_with_grounding_check():
    mock_retriever = MagicMock()
    mock_llm = FakeListChatModel(responses=["FastAPI is an asynchronous Python framework [1]."])

    doc1 = Document(page_content="FastAPI is an async web framework.", metadata={"source": "docs.md"})
    mock_retriever.invoke.return_value = [doc1]

    pipeline = CRAGPipeline(
        retriever=mock_retriever, llm=mock_llm,
        relevance_threshold=0.5, enable_grounding_check=True,
    )

    pipeline.grader.grade_documents = MagicMock(
        return_value=GradingResult(
            graded_docs=[GradedDocument(doc1, True, 0.9)],
            relevant_docs=[doc1],
            relevance_ratio=1.0,
            confidence_level="HIGH",
        )
    )

    pipeline.grounding_checker.check = MagicMock(
        return_value=GroundingReport(
            is_grounded=True,
            faithfulness_score=1.0,
            hallucinated_claims=[],
            summary="All claims grounded in documentation.",
        )
    )

    output = pipeline.query("What is FastAPI?")

    assert output.grounding_report is not None
    assert output.grounding_report.is_grounded is True
    assert output.grounding_report.faithfulness_score == 1.0
