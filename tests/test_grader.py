"""Tests for DocumentGrader in Corrective RAG."""

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from minirag.grader import DocumentGrader, GradedDocument, GradingResult


def test_grade_document_relevant():
    mock_llm = FakeListChatModel(
        responses=['{"is_relevant": true, "score": 0.95, "reason": "Mentions attention mechanisms directly."}']
    )

    grader = DocumentGrader(mock_llm)
    doc = Document(page_content="Attention mechanisms allow models to focus on specific tokens.")
    result = grader.grade_document(doc, "What are attention mechanisms?")

    assert isinstance(result, GradedDocument)
    assert result.is_relevant is True
    assert result.score == 0.95
    assert "attention" in result.reason.lower()


def test_grade_document_irrelevant_with_markdown_fence():
    mock_llm = FakeListChatModel(
        responses=['```json\n{"is_relevant": false, "score": 0.1, "reason": "Discusses cooking recipes."}\n```']
    )

    grader = DocumentGrader(mock_llm)
    doc = Document(page_content="To bake a cake, preheat the oven to 350 degrees.")
    result = grader.grade_document(doc, "What is quantum computing?")

    assert result.is_relevant is False
    assert result.score == 0.1


def test_grade_documents_aggregation():
    mock_llm = FakeListChatModel(
        responses=[
            '{"is_relevant": true, "score": 0.9, "reason": "Match"}',
            '{"is_relevant": false, "score": 0.1, "reason": "No match"}',
            '{"is_relevant": true, "score": 0.8, "reason": "Match"}',
        ]
    )

    grader = DocumentGrader(mock_llm)
    docs = [
        Document(page_content="Doc 1 content"),
        Document(page_content="Doc 2 content"),
        Document(page_content="Doc 3 content"),
    ]

    res = grader.grade_documents(docs, "test question", high_threshold=0.6, partial_threshold=0.3)
    assert isinstance(res, GradingResult)
    assert len(res.graded_docs) == 3
    assert len(res.relevant_docs) == 2
    assert abs(res.relevance_ratio - (2 / 3)) < 1e-4
    assert res.confidence_level == "HIGH"
