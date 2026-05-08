"""Shared pytest fixtures.

We aggressively stub the LLM and the embedding model so tests are:
    * fast (no model download, no Ollama dependency),
    * deterministic (no randomness, no network),
    * self-contained (each test gets a temp Chroma directory).
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Sequence

import pytest

# Ensure the project root is importable regardless of where pytest is launched.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Per-test isolated config


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the app at a clean temp data + vector_store folder."""
    data_dir = tmp_path / "data"
    vector_store_dir = tmp_path / "vector_store"
    data_dir.mkdir()
    vector_store_dir.mkdir()

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("VECTOR_STORE_DIR", str(vector_store_dir))
    monkeypatch.setenv("COLLECTION_NAME", "test_collection")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:0")  # not reachable

    # Reset cached singletons between tests.
    from app import config as _config
    from app import vectorstore as _vs
    from app import embeddings as _emb
    _config.get_settings.cache_clear()
    _vs.reset_vector_store_singleton()
    _emb._load_model.cache_clear()  # type: ignore[attr-defined]

    yield

    # Close Chroma's SQLite handles before tmp_path is removed.
    # Otherwise pytest's rmtree loops on the still-open file on some platforms.
    try:
        import chromadb  # noqa: PLC0415
        chromadb.api.client.SharedSystemClient.clear_system_cache()
    except Exception:  # noqa: BLE001
        pass

    _config.get_settings.cache_clear()
    _vs.reset_vector_store_singleton()
    _emb._load_model.cache_clear()  # type: ignore[attr-defined]

    import gc  # noqa: PLC0415
    gc.collect()


# ---------------------------------------------------------------------------
# Embedding stub


def _hash_embedding(text: str, dim: int = 32) -> list[float]:
    """Deterministic pseudo-embedding from a SHA hash.

    Uses the bytes of repeated SHA-256 digests so we get stable vectors
    without loading sentence-transformers.
    """
    digest = b""
    seed = text.encode("utf-8")
    while len(digest) < dim * 4:
        seed = hashlib.sha256(seed).digest()
        digest += seed
    raw = digest[: dim * 4]
    floats = [
        int.from_bytes(raw[i : i + 4], "little") / (2**32 - 1)
        for i in range(0, dim * 4, 4)
    ]
    # Normalise so cosine distance behaves.
    norm = sum(f * f for f in floats) ** 0.5 or 1.0
    return [f / norm for f in floats]


@pytest.fixture(autouse=True)
def _stub_embeddings(monkeypatch: pytest.MonkeyPatch):
    from app import embeddings

    def fake_embed_texts(texts: Sequence[str]) -> list[list[float]]:
        return [_hash_embedding(t) for t in texts]

    def fake_embed_query(query: str) -> list[float]:
        return _hash_embedding(query)

    def fake_embedding_dimension() -> int:
        return 32

    monkeypatch.setattr(embeddings, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(embeddings, "embed_query", fake_embed_query)
    monkeypatch.setattr(embeddings, "embedding_dimension", fake_embedding_dimension)

    # Patch the symbols where they are imported, too.
    from app import ingestion, rag
    monkeypatch.setattr(ingestion, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(rag, "embed_query", fake_embed_query)


# ---------------------------------------------------------------------------
# LLM stub


@pytest.fixture
def stub_llm(monkeypatch: pytest.MonkeyPatch):
    """Replace the chat() function with a configurable canned response."""
    from app import agent, llm, rag

    state = {
        "answer": "stubbed answer",
        "intent_label": "knowledge",
    }

    def fake_chat(system_prompt: str, user_prompt: str, **kwargs):
        # If the system prompt is the router prompt, return the configured intent.
        if "intent classifier" in system_prompt:
            return llm.LLMResponse(
                text=state["intent_label"],
                model="stub",
                prompt_tokens=0,
                completion_tokens=0,
            )
        return llm.LLMResponse(
            text=state["answer"],
            model="stub",
            prompt_tokens=0,
            completion_tokens=0,
        )

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(rag, "chat", fake_chat)
    monkeypatch.setattr(agent, "chat", fake_chat)

    return state
