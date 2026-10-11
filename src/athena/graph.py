"""LangGraph construction for self-correcting RAG workflow."""

from langgraph.graph import END, START, StateGraph

from athena.nodes import (
    generate_answer,
    grade_documents,
    retrieve_documents,
    rewrite_question,
)
from athena.state import RAGState


def should_rewrite(state: RAGState) -> str:
    """Route: generate if relevant, rewrite if not (and under max attempts)."""
    if state.get("all_relevant"):
        return "generate"
    if state.get("rewrite_count", 0) >= state.get("max_rewrites", 2):
        return "generate"  # Give up gracefully
    return "rewrite"


def build_graph():
    """Assemble and compile the self-correcting RAG state graph."""
    builder = StateGraph(RAGState)

    builder.add_node("retrieve", retrieve_documents)
    builder.add_node("grade", grade_documents)
    builder.add_node("rewrite", rewrite_question)
    builder.add_node("generate", generate_answer)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")

    builder.add_conditional_edges(
        "grade",
        should_rewrite,
        {
            "generate": "generate",
            "rewrite": "rewrite",
        },
    )

    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("generate", END)

    return builder.compile()
