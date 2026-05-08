"""Sentence-transformers embedding wrapper.

Wrapped so we can swap models or providers later without touching the
rest of the codebase. The model is loaded lazily on first use so module
import stays fast and so tests can stub the embedding functions without
needing sentence-transformers to be installed.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

from app.config import get_settings
from app.logger import get_logger, log_event

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_model() -> Any:
    """Load the sentence-transformers model lazily."""
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    settings = get_settings()
    log_event(log, "embeddings.loading", model=settings.embedding_model)
    model = SentenceTransformer(settings.embedding_model)
    log_event(
        log,
        "embeddings.loaded",
        model=settings.embedding_model,
        dim=model.get_sentence_embedding_dimension(),
    )
    return model


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns one vector per input text."""
    if not texts:
        return []
    model = _load_model()
    vectors = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]


def embedding_dimension() -> int:
    """Return the dimensionality of the configured embedding model."""
    return _load_model().get_sentence_embedding_dimension()
