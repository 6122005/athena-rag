"""Central LLM factory supporting OpenAI, 100% Free Groq, and Streamlit Cloud secrets."""

import os
from langchain_openai import ChatOpenAI


def get_config(key: str, default: str | None = None) -> str | None:
    """
    Unified configuration resolver.
    Checks environment variables first, then falls back to Streamlit secrets (st.secrets)
    when running on Streamlit Community Cloud.
    """
    val = os.getenv(key)
    if val:
        return val

    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    return default


def get_chat_model(temperature: float = 0):
    """
    Initialize and return the active language model.
    Seamlessly adapts to:
    1. Groq (100% Free via Llama / Qwen on Groq Cloud)
    2. OpenAI (GPT-4o-mini)
    """
    groq_key = get_config("GROQ_API_KEY")
    openai_key = get_config("OPENAI_API_KEY")
    provider = (get_config("LLM_PROVIDER") or "").lower()

    # Prioritize Groq if provider is 'groq' or if GROQ_API_KEY is present
    if provider == "groq" or (groq_key and provider != "openai"):
        model_name = get_config("GROQ_MODEL", "qwen/qwen3.8-27b")
        return ChatOpenAI(
            model=model_name,
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=temperature,
        )

    # OpenAI configuration
    api_key = openai_key or "sk-placeholder-key-for-initialization"
    return ChatOpenAI(
        model=get_config("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=api_key,
        temperature=temperature,
    )
