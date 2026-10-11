"""Structured JSON logging for every RAG transition."""

import json
import logging
import os
import sys


def setup_logger(name: str = "athena") -> logging.Logger:
    """Set up and configure a structured logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, event: str, **kwargs) -> None:
    """Emit a single structured JSON event log line."""
    logger.info(json.dumps({"event": event, **kwargs}, default=str))
