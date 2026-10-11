"""CLI entry point for local testing."""

import sys
from athena.graph import build_graph


def run(question: str) -> None:
    """Execute the RAG workflow for a given question and display formatted results."""
    graph = build_graph()
    result = graph.invoke({
        "question": question,
        "retrieval_query": question,
        "rewrite_count": 0,
        "max_rewrites": 2,
        "trace": [],
    })

    print("=" * 60)
    print(f"Question: {question}")
    print("-" * 60)
    print(f"Answer: {result.get('answer')}")
    print(f"Sources: {result.get('sources', [])}")
    print(f"Rewrites: {result.get('rewrite_count', 0)}")
    print("-" * 60)
    print("Trace:")
    for entry in result.get("trace", []):
        print(f"  {entry}")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is retrieval augmented generation?"
    run(q)
