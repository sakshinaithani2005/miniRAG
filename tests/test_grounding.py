"""Tests for GroundingChecker module."""

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from minirag.grounding import GroundingChecker, GroundingReport


def test_grounding_checker_grounded_answer():
    mock_llm = FakeListChatModel(
        responses=['{"is_grounded": true, "faithfulness_score": 1.0, "hallucinated_claims": [], "summary": "Fully verified."}']
    )

    checker = GroundingChecker(mock_llm)
    docs = [Document(page_content="The model was trained on 100M tokens.")]
    report = checker.check(
        "The model used 100M tokens for training [1].", docs, "How was the model trained?"
    )

    assert isinstance(report, GroundingReport)
    assert report.is_grounded is True
    assert report.faithfulness_score == 1.0
    assert len(report.hallucinated_claims) == 0


def test_grounding_checker_hallucinated_answer():
    mock_llm = FakeListChatModel(
        responses=['{"is_grounded": false, "faithfulness_score": 0.5, "hallucinated_claims": ["Trained for 10 years"], "summary": "Detected hallucinated claim."}']
    )

    checker = GroundingChecker(mock_llm)
    docs = [Document(page_content="The model was trained on 100M tokens.")]
    report = checker.check(
        "The model was trained on 100M tokens and trained for 10 years [1].", docs, "Details?"
    )

    assert report.is_grounded is False
    assert report.faithfulness_score == 0.5
    assert len(report.hallucinated_claims) == 1
