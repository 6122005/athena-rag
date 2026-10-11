"""Unit tests for Chroma vector store and client initialization resilience."""

import os
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from athena.tools.vectorstore import (
    get_chroma_client,
    get_vectorstore,
    reset_vectorstore,
    build_vectorstore,
    get_retriever,
)


def test_get_chroma_client_persistent(tmp_path):
    """Verify get_chroma_client creates a persistent client."""
    reset_vectorstore()
    client = get_chroma_client(persist_dir=str(tmp_path / "test_db"))
    assert client is not None
    assert hasattr(client, "heartbeat")


def test_get_chroma_client_fallback_on_error(tmp_path):
    """Verify get_chroma_client falls back to EphemeralClient on persistent initialization error."""
    reset_vectorstore()
    with patch("chromadb.PersistentClient", side_effect=Exception("Disk permission denied")):
        client = get_chroma_client(persist_dir=str(tmp_path / "fail_db"))
        assert client is not None
        assert hasattr(client, "heartbeat")


def test_build_vectorstore_and_retrieve(tmp_path):
    """Verify building vectorstore and retrieving documents."""
    reset_vectorstore()
    test_dir = str(tmp_path / "test_store")
    docs = [
        {"content": "Machine learning is a field of artificial intelligence.", "source": "ml.txt"},
        {"content": "Quantum computing uses qubits for computation.", "source": "qc.txt"},
    ]

    store = build_vectorstore(docs, persist_dir=test_dir)
    assert store is not None

    retriever = get_retriever(k=1, persist_dir=test_dir)
    results = retriever.invoke("artificial intelligence")
    assert len(results) > 0
    assert "artificial intelligence" in results[0].page_content.lower()
