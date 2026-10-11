"""End-to-end graph tests with mocked LLM and retriever calls."""

from unittest.mock import MagicMock, patch

from athena.graph import build_graph


@patch("athena.nodes.retrieve.get_retriever")
@patch("athena.nodes.grade.grade_model")
@patch("athena.nodes.generate.generate_model")
def test_full_graph_relevant_path(mock_generate, mock_grade, mock_get_retriever):
    """Test full happy path where retrieved documents pass grading and generate answer."""
    mock_retriever = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "RAG combines retrieval with generation."
    mock_doc.metadata = {"source": "rag_intro.md"}
    mock_retriever.invoke.return_value = [mock_doc]
    mock_get_retriever.return_value = mock_retriever

    mock_grade.invoke.return_value = MagicMock(content='{"binary_score": "yes"}')
    mock_generate.invoke.return_value = MagicMock(content="RAG combines retrieval with generation.")

    graph = build_graph()
    result = graph.invoke({
        "question": "What is RAG?",
        "retrieval_query": "What is RAG?",
        "rewrite_count": 0,
        "max_rewrites": 2,
        "trace": [],
    })

    assert result["answer"] is not None
    assert result["rewrite_count"] == 0
    assert "rag_intro.md" in result["sources"]


@patch("athena.nodes.retrieve.get_retriever")
@patch("athena.nodes.grade.grade_model")
@patch("athena.nodes.rewrite.rewrite_model")
@patch("athena.nodes.generate.generate_model")
def test_full_graph_rewrite_loop_path(mock_generate, mock_rewrite, mock_grade, mock_get_retriever):
    """Test self-correction loop where initial retrieval fails, gets rewritten, and succeeds."""
    mock_retriever = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "Detailed RAG architecture and retrieval mechanisms."
    mock_doc.metadata = {"source": "rag_intro.md"}
    mock_retriever.invoke.return_value = [mock_doc]
    mock_get_retriever.return_value = mock_retriever

    # Grade fails first, then passes on second attempt
    mock_grade.invoke.side_effect = [
        MagicMock(content='{"binary_score": "no"}'),
        MagicMock(content='{"binary_score": "yes"}'),
    ]
    mock_rewrite.invoke.return_value = MagicMock(content="What is retrieval augmented generation architecture?")
    mock_generate.invoke.return_value = MagicMock(content="RAG combines retriever and generator components.")

    graph = build_graph()
    result = graph.invoke({
        "question": "Tell me about it",
        "retrieval_query": "Tell me about it",
        "rewrite_count": 0,
        "max_rewrites": 2,
        "trace": [],
    })

    assert result["rewrite_count"] == 1
    assert result["answer"] == "RAG combines retriever and generator components."


@patch("athena.nodes.retrieve.get_retriever")
@patch("athena.nodes.grade.grade_model")
@patch("athena.nodes.rewrite.rewrite_model")
@patch("athena.nodes.generate.generate_model")
def test_full_graph_max_rewrites_exceeded(mock_generate, mock_rewrite, mock_grade, mock_get_retriever):
    """Test that graph gracefully stops rewriting and falls back when max_rewrites is reached."""
    mock_retriever = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "Completely irrelevant content."
    mock_doc.metadata = {"source": "unrelated.md"}
    mock_retriever.invoke.return_value = [mock_doc]
    mock_get_retriever.return_value = mock_retriever

    mock_grade.invoke.return_value = MagicMock(content='{"binary_score": "no"}')
    mock_rewrite.invoke.return_value = MagicMock(content="Attempted rewrite query")
    mock_generate.invoke.return_value = MagicMock(content="Fallback answer")

    graph = build_graph()
    result = graph.invoke({
        "question": "Completely unknown topic",
        "retrieval_query": "Completely unknown topic",
        "rewrite_count": 0,
        "max_rewrites": 1,
        "trace": [],
    })

    assert result["rewrite_count"] == 1
    assert result["answer"] is not None
