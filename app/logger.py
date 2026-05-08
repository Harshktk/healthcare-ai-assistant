"""Structured logging configuration.

We use the standard library ``logging`` module with a JSON-friendly
format. Each request gets a request id propagated via ``extra`` so we
can correlate the ingestion → retrieval → generation flow in production.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from app.config import get_settings

_LOGGER_NAME = "healthcare_ai"
_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "%(message)s"
)


def _configure_root_logger() -> None:
    settings = get_settings()
    root = logging.getLogger()

    # Avoid double-configuration if uvicorn or pytest already attached handlers.
    if getattr(root, "_healthcare_configured", False):
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT))

    root.handlers = [handler]
    root.setLevel(settings.log_level.upper())

    # Keep noisy third-party libraries at WARNING.
    for noisy in ("httpx", "httpcore", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    setattr(root, "_healthcare_configured", True)


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger with the project's standard configuration."""
    _configure_root_logger()
    return logging.getLogger(name or _LOGGER_NAME)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Emit a structured INFO log line.

    Example:
        log_event(log, "retrieval.completed", top_k=4, top_distance=0.31)

    Produces:
        ... | INFO | healthcare_ai | retrieval.completed top_k=4 top_distance=0.31
    """
    if fields:
        body = " ".join(f"{k}={v}" for k, v in fields.items())
        logger.info("%s %s", event, body)
    else:
        logger.info(event)
