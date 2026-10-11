"""Tests for document ingestion utilities."""

from unittest.mock import MagicMock, patch
import pytest

from athena.ingestion import (
    chunk_text,
    ingest_documents,
    load_document_from_file,
    load_documents_from_directory,
    load_documents_from_json,
    validate_document,
)


def test_chunk_text_short():
    """Verify short text is returned in a single chunk."""
    text = "Short text under chunk size."
    chunks = chunk_text(text, chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_long():
    """Verify long text is broken into multiple chunks with overlap."""
    text = "word " * 100
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 1


def test_validate_document_valid():
    """Verify document validation succeeds on valid document."""
    doc = {"content": "Sample content", "source": "sample.md"}
    result = validate_document(doc)
    assert result["content"] == "Sample content"
    assert result["source"] == "sample.md"


def test_validate_document_invalid():
    """Verify document validation raises error when content is missing."""
    with pytest.raises(ValueError):
        validate_document({"source": "sample.md"})


def test_load_document_from_file_text(tmp_path):
    """Verify loading plain text document from disk."""
    sample_txt = tmp_path / "policy.txt"
    sample_txt.write_text("Company refund policy guidelines.", encoding="utf-8")

    docs = load_document_from_file(sample_txt)
    assert len(docs) == 1
    assert docs[0]["source"] == "policy.txt"
    assert docs[0]["content"] == "Company refund policy guidelines."


def test_load_documents_from_directory(tmp_path):
    """Verify loading multiple documents from directory."""
    (tmp_path / "doc1.txt").write_text("First document text.", encoding="utf-8")
    (tmp_path / "doc2.md").write_text("Second markdown doc.", encoding="utf-8")

    docs = load_documents_from_directory(tmp_path)
    assert len(docs) == 2


@patch("athena.ingestion.build_vectorstore")
def test_ingest_documents(mock_build_vs):
    """Verify ingest_documents processes and forwards docs to vector store."""
    docs = [{"content": "Doc 1 content", "source": "doc1.md"}]
    count = ingest_documents(docs, persist_dir="./test_chroma_db", auto_chunk=False)
    assert count == 1
    mock_build_vs.assert_called_once()
