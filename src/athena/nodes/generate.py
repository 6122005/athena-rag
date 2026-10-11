"""Answer generation node: grounded response from verified context."""

from unittest.mock import Mock
from langchain_core.prompts import ChatPromptTemplate

from athena.model import get_chat_model
from athena.observability import log_event, setup_logger
from athena.state import RAGState

logger = setup_logger("athena.generate")

GENERATE_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below. If the context does not contain the answer, say so explicitly.

Context:
{context}

Question: {question}

Provide a concise, direct answer."""
)

generate_model = get_chat_model(temperature=0)


def _resolve_model():
    global generate_model
    if isinstance(generate_model, Mock):
        return generate_model
    return get_chat_model(temperature=0)


def generate_answer(state: RAGState) -> dict:
    """Generate grounded answer strictly adhering to retrieved and graded context."""
    question = state["question"]
    documents = state.get("documents", [])
    model = _resolve_model()

    # Keep only documents that passed grading
    relevant_docs = [
        doc for doc, score in zip(documents, state.get("relevance_scores", []))
        if score == "yes"
    ]

    if not relevant_docs:
        return {
            "answer": "I could not find relevant information in the knowledge base to answer this question.",
            "sources": [],
            "trace": [{"node": "generate", "action": "no_relevant_docs"}],
        }

    context = "\n\n---\n\n".join(d["content"] for d in relevant_docs)
    sources = [d["source"] for d in relevant_docs]

    prompt = GENERATE_PROMPT.format(context=context[:6000], question=question)

    try:
        response = model.invoke([{"role": "user", "content": prompt}])
        answer = response.content
    except Exception as e:
        err_msg = str(e).lower()
        if "insufficient_quota" in err_msg or "credit_balance_exhausted" in err_msg or "429" in err_msg:
            answer = (
                "⚠️ **OpenAI Credit Exhausted!**\n\n"
                "Please select **Groq (100% Free)** in the sidebar or set `LLM_PROVIDER=groq` in `.env`."
            )
        else:
            answer = f"Error generating answer: {str(e)}"

    log_event(logger, "generate", sources=sources, answer_length=len(answer))

    return {
        "answer": answer,
        "sources": list(set(sources)),
        "trace": [{"node": "generate", "sources": sources}],
    }
