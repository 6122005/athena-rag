"""Vector store abstraction. ChromaDB with automatic fallback and batch ingestion."""

import logging
import os
from typing import Callable
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from athena.model import get_config

logger = logging.getLogger("athena.vectorstore")

_embeddings = None
_chroma_client = None
_vectorstore = None


class LocalDefaultEmbeddings(Embeddings):
    """High-speed local dense embeddings powered by Chroma's ONNX all-MiniLM-L6-v2."""

    def __init__(self):
        import chromadb.utils.embedding_functions as ef
        self._ef = ef.DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Process in smaller internal batches to avoid CPU/memory stalls
        batch_size = 32
        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            results.extend(self._ef(chunk))
        return results

    def embed_query(self, text: str) -> list[float]:
        return self._ef([text])[0]


class FallbackEmbeddings(Embeddings):
    """
    Attempts OpenAI embeddings first.
    If the OpenAI project or key does not have access to embedding models (403/model_not_found),
    it transparently falls back to local ONNX embeddings without crashing.
    """

    def __init__(self, openai_embeddings: OpenAIEmbeddings, local_embeddings: LocalDefaultEmbeddings):
        self._openai = openai_embeddings
        self._local = local_embeddings
        provider = (get_config("EMBEDDING_PROVIDER") or "").lower()
        self._use_local = (provider == "local")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not self._use_local:
            try:
                return self._openai.embed_documents(texts)
            except Exception as e:
                logger.warning(
                    f"OpenAI embedding unavailable ({e}). Automatically falling back to local ONNX embeddings."
                )
                self._use_local = True
        return self._local.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if not self._use_local:
            try:
                return self._openai.embed_query(text)
            except Exception as e:
                logger.warning(
                    f"OpenAI embedding unavailable ({e}). Automatically falling back to local ONNX embeddings."
                )
                self._use_local = True
        return self._local.embed_query(text)


def _get_embeddings():
    """Lazily initialize and return embeddings with self-healing local fallback."""
    global _embeddings
    if _embeddings is None:
        api_key = get_config("OPENAI_API_KEY", "sk-placeholder-key-for-local")
        openai_emb = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
        local_emb = LocalDefaultEmbeddings()
        _embeddings = FallbackEmbeddings(openai_emb, local_emb)
    return _embeddings


def get_chroma_client(persist_dir: str = "./chroma_db"):
    """
    Get or create a robust ChromaDB client.
    Attempts PersistentClient with absolute directory first.
    If persistent initialization fails (e.g. tenant error, disk permissions on Streamlit Cloud),
    transparently falls back to EphemeralClient (in-memory) so document ingestion never crashes.
    """
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client

    import chromadb
    from chromadb.config import Settings

    abs_path = os.path.abspath(persist_dir)
    try:
        os.makedirs(abs_path, exist_ok=True)
        settings = Settings(anonymized_telemetry=False, allow_reset=True)
        _chroma_client = chromadb.PersistentClient(path=abs_path, settings=settings)
        _chroma_client.heartbeat()
        logger.info(f"Initialized ChromaDB PersistentClient at {abs_path}")
    except Exception as exc:
        logger.warning(
            f"PersistentClient initialization encountered '{exc}'. "
            "Falling back to in-memory EphemeralClient for cloud resilience."
        )
        settings = Settings(anonymized_telemetry=False, allow_reset=True)
        _chroma_client = chromadb.EphemeralClient(settings=settings)

    return _chroma_client


def get_vectorstore(persist_dir: str = "./chroma_db"):
    """Get or initialize singleton vector store. Falls back to InMemoryVectorStore if needed."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    # 1. Attempt Chroma with explicit client
    try:
        client = get_chroma_client(persist_dir=persist_dir)
        _vectorstore = Chroma(
            client=client,
            collection_name="athena_knowledge",
            embedding_function=_get_embeddings(),
        )
        return _vectorstore
    except Exception as exc:
        logger.warning(
            f"Chroma store initialization encountered '{exc}'. "
            "Falling back to InMemoryVectorStore for cloud resilience."
        )

    # 2. Resilient fallback to LangChain's InMemoryVectorStore
    try:
        from langchain_core.vectorstores import InMemoryVectorStore
        _vectorstore = InMemoryVectorStore(embedding=_get_embeddings())
        return _vectorstore
    except Exception as exc2:
        logger.error(f"Failed to initialize vector store: {exc2}")
        raise exc2


def reset_vectorstore():
    """Reset the singleton vector store and Chroma client."""
    global _vectorstore, _chroma_client
    _vectorstore = None
    _chroma_client = None


def build_vectorstore(
    docs: list[dict],
    persist_dir: str = "./chroma_db",
    callback: Callable[[int, int], None] | None = None,
    batch_size: int = 20,
):
    """Build or update a vector store with batch indexing and live progress callback."""
    global _vectorstore
    store = get_vectorstore(persist_dir=persist_dir)

    documents = [
        Document(page_content=d["content"], metadata={"source": d["source"], **d.get("metadata", {})})
        for d in docs
    ]

    total = len(documents)
    if total == 0:
        return store

    try:
        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            store.add_documents(batch)
            if callback:
                callback(min(i + batch_size, total), total)
    except Exception as exc:
        logger.warning(
            f"Failed to add documents to primary store ({exc}). Switching to InMemoryVectorStore fallback."
        )
        from langchain_core.vectorstores import InMemoryVectorStore
        _vectorstore = InMemoryVectorStore(embedding=_get_embeddings())
        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            _vectorstore.add_documents(batch)
            if callback:
                callback(min(i + batch_size, total), total)
        return _vectorstore

    return store


def get_retriever(k: int = 4, persist_dir: str = "./chroma_db"):
    """Return a similarity retriever. Builds empty store if none exists."""
    store = get_vectorstore(persist_dir=persist_dir)
    return store.as_retriever(search_type="similarity", search_kwargs={"k": k})

