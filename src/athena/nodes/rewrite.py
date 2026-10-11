"""Query rewriting node: reformulate failed retrieval queries."""

from unittest.mock import Mock
from langchain_core.prompts import ChatPromptTemplate

from athena.model import get_chat_model
from athena.observability import log_event, setup_logger
from athena.state import RAGState

logger = setup_logger("athena.rewrite")

REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """The original question did not retrieve relevant documents. Reformulate it into a clearer, more specific search query.

Original question: {question}

Respond with ONLY the improved query. No explanation."""
)

rewrite_model = get_chat_model(temperature=0)


def _resolve_model():
    global rewrite_model
    if isinstance(rewrite_model, Mock):
        return rewrite_model
    return get_chat_model(temperature=0)


def rewrite_question(state: RAGState) -> dict:
    """Reformulate query to enhance retrieval recall on retry."""
    question = state["question"]
    rewrite_count = state.get("rewrite_count", 0)
    model = _resolve_model()

    prompt = REWRITE_PROMPT.format(question=question)
    try:
        response = model.invoke([{"role": "user", "content": prompt}])
        new_query = response.content.strip()
    except Exception as e:
        logger.warning(f"Query rewrite encountered error: {e}. Preserving original question.")
        new_query = question

    log_event(
        logger,
        "rewrite",
        original=question[:100],
        rewritten=new_query[:100],
        attempt=rewrite_count + 1,
    )

    return {
        "retrieval_query": new_query,
        "rewrite_count": rewrite_count + 1,
        "trace": [{
            "node": "rewrite",
            "attempt": rewrite_count + 1,
            "new_query": new_query[:100],
        }],
    }
