"""Vector store abstraction. ChromaDB with automatic fallback between OpenAI and Local Embeddings."""

import logging
import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from athena.model import get_config

logger = logging.getLogger("athena.vectorstore")

_embeddings = None
_vectorstore = None


class LocalDefaultEmbeddings(Embeddings):
    """High-speed local dense embeddings powered by Chroma's ONNX all-MiniLM-L6-v2."""

    def __init__(self):
        import chromadb.utils.embedding_functions as ef
        self._ef = ef.DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._ef(texts)

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


def build_vectorstore(docs: list[dict], persist_dir: str = "./chroma_db"):
    """Build or load a Chroma vector store from raw document dicts."""
    global _vectorstore

    documents = [
        Document(page_content=d["content"], metadata={"source": d["source"], **d.get("metadata", {})})
        for d in docs
    ]

    _vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=_get_embeddings(),
        persist_directory=persist_dir,
        collection_name="athena_knowledge",
    )
    return _vectorstore


def get_retriever(k: int = 4, persist_dir: str = "./chroma_db"):
    """Return a similarity retriever. Builds empty store if none exists."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            embedding_function=_get_embeddings(),
            persist_directory=persist_dir,
            collection_name="athena_knowledge",
        )
    return _vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
