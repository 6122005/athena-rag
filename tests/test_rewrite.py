"""Rewrite node tests."""

from unittest.mock import MagicMock, patch

from athena.nodes.rewrite import rewrite_question


@patch("athena.nodes.rewrite.rewrite_model")
def test_rewrite_increments_count(mock_model):
    """Test that rewrite_question increments rewrite_count and updates retrieval_query."""
    mock_model.invoke.return_value = MagicMock(content="What is Retrieval Augmented Generation?")
    state = {
        "question": "RAG?",
        "rewrite_count": 0,
        "trace": [],
    }
    result = rewrite_question(state)
    assert result["rewrite_count"] == 1
    assert "Retrieval Augmented Generation" in result["retrieval_query"]
    assert len(result["trace"]) == 1
    assert result["trace"][0]["node"] == "rewrite"
