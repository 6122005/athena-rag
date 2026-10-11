"""State schema definition for the Athena RAG LangGraph workflow."""

import operator
from typing import Annotated, TypedDict


class RAGState(TypedDict):
    """Represents the complete state of the self-correcting RAG graph."""

    # Input
    question: str

    # Retrieval
    documents: list[dict]
    retrieval_query: str

    # Grading
    relevance_scores: list[str]
    all_relevant: bool

    # Loop control
    rewrite_count: int
    max_rewrites: int

    # Output
    answer: str | None
    sources: list[str]

    # Audit
    trace: Annotated[list[dict], operator.add]
