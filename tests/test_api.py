"""API endpoint tests for Athena RAG FastAPI service."""

import io
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from athena.api import app

client = TestClient(app)


def test_health_endpoint():
    """Verify health readiness probe returns ok status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("athena.api.graph")
def test_query_endpoint_success(mock_graph):
    """Verify /query endpoint invokes graph and formats response."""
    mock_graph.invoke.return_value = {
        "answer": "Retrieval-Augmented Generation anchors LLM answers in facts.",
        "sources": ["rag_intro.md"],
        "rewrite_count": 0,
        "trace": [{"node": "retrieve"}, {"node": "grade"}, {"node": "generate"}],
    }

    response = client.post("/query", json={"question": "What is RAG?", "max_rewrites": 2})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Retrieval-Augmented Generation anchors LLM answers in facts."
    assert data["sources"] == ["rag_intro.md"]
    assert data["rewrites"] == 0
    assert len(data["trace"]) == 3


@patch("athena.api.build_vectorstore")
def test_ingest_endpoint_success(mock_build_vs):
    """Verify /ingest endpoint invokes build_vectorstore."""
    sample_docs = [
        {"source": "test.md", "content": "Sample content for testing ingestion."}
    ]
    response = client.post("/ingest", json={"documents": sample_docs})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ingested"
    assert data["count"] == 1
    mock_build_vs.assert_called_once()


def test_upload_pdf_invalid_extension():
    """Verify upload-pdf rejects non-pdf files with 400 Bad Request."""
    file_content = b"Some plain text content"
    files = {"file": ("test.txt", io.BytesIO(file_content), "text/plain")}
    response = client.post("/upload-pdf", files=files)
    assert response.status_code == 400
    assert "must be a PDF" in response.json()["detail"]


@patch("athena.api.ingest_documents")
@patch("athena.api.extract_pdf_pages")
def test_upload_pdf_success(mock_extract, mock_ingest):
    """Verify upload-pdf extracts and ingests valid PDF file."""
    mock_extract.return_value = [
        {"source": "sample.pdf (Page 1)", "content": "Extracted page text"}
    ]
    mock_ingest.return_value = 1

    file_content = b"%PDF-1.4 dummy pdf bytes"
    files = {"file": ("sample.pdf", io.BytesIO(file_content), "application/pdf")}
    response = client.post("/upload-pdf", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "sample.pdf"
    assert data["chunks_ingested"] == 1
