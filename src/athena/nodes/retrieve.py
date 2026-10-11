"""Retrieval node: fetch relevant documents from vector store."""

from athena.observability import log_event, setup_logger
from athena.state import RAGState
from athena.tools import get_retriever

logger = setup_logger("athena.retrieve")


def retrieve_documents(state: RAGState) -> dict:
    """Retrieve candidate documents matching the question or rewritten query."""
    query = state.get("retrieval_query") or state["question"]
    retriever = get_retriever(k=4)

    try:
        docs = retriever.invoke(query)
    except Exception as exc:
        logger.warning(f"Retrieval query encountered issue: {exc}")
        docs = []

    doc_dicts = [
        {"content": d.page_content, "source": d.metadata.get("source", "unknown")}
        for d in docs
    ]

    log_event(logger, "retrieve", query=query[:100], count=len(doc_dicts))

    return {
        "documents": doc_dicts,
        "retrieval_query": query,
        "trace": [{"node": "retrieve", "query": query[:100], "count": len(doc_dicts)}],
    }
