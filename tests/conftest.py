"""Pytest global configuration and fixtures."""

import os
import pytest

# Ensure a mock OPENAI_API_KEY is available during test collection and execution
os.environ.setdefault("OPENAI_API_KEY", "sk-mock-key-for-testing-athena-rag")
os.environ.setdefault("LOG_LEVEL", "WARNING")
