"""Grade node tests."""

from unittest.mock import MagicMock, patch

from athena.nodes.grade import grade_documents


@patch("athena.nodes.grade.grade_model")
def test_relevant_document(mock_model):
    """Test that a document graded as relevant marks all_relevant as True."""
    mock_model.invoke.return_value = MagicMock(content='{"binary_score": "yes"}')
    state = {
        "question": "What is RAG?",
        "documents": [{"content": "RAG combines retrieval and generation.", "source": "test"}],
        "trace": [],
    }
    result = grade_documents(state)
    assert result["all_relevant"] is True
    assert result["relevance_scores"] == ["yes"]


@patch("athena.nodes.grade.grade_model")
def test_irrelevant_document(mock_model):
    """Test that an irrelevant document marks all_relevant as False."""
    mock_model.invoke.return_value = MagicMock(content='{"binary_score": "no"}')
    state = {
        "question": "What is RAG?",
        "documents": [{"content": "The weather is sunny.", "source": "test"}],
        "trace": [],
    }
    result = grade_documents(state)
    assert result["all_relevant"] is False
    assert result["relevance_scores"] == ["no"]


@patch("athena.nodes.grade.grade_model")
def test_invalid_json_fallback(mock_model):
    """Test that invalid model JSON output falls back gracefully to 'no'."""
    mock_model.invoke.return_value = MagicMock(content="invalid-json")
    state = {
        "question": "What is RAG?",
        "documents": [{"content": "Some doc content.", "source": "test"}],
        "trace": [],
    }
    result = grade_documents(state)
    assert result["all_relevant"] is False
    assert result["relevance_scores"] == ["no"]
