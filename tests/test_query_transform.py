"""Tests for QueryTransformer module."""

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from minirag.query_transform import QueryTransformer


def test_transform_for_web():
    mock_llm = FakeListChatModel(responses=["latest transformer architecture benchmarks 2025"])

    transformer = QueryTransformer(mock_llm)
    query = transformer.transform_for_web(
        "what are the latest benchmarks?", context_hints="Missing 2025 papers"
    )

    assert query == "latest transformer architecture benchmarks 2025"


def test_decompose_query():
    mock_llm = FakeListChatModel(
        responses=['["What is BERT?", "What is GPT?", "How do they compare?"]']
    )

    transformer = QueryTransformer(mock_llm)
    sub_queries = transformer.decompose("Compare BERT and GPT models.")

    assert len(sub_queries) == 3
    assert sub_queries[0] == "What is BERT?"
