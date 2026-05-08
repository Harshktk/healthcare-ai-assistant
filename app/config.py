"""Application configuration loaded from environment variables.

We use ``pydantic-settings`` so every setting is type-checked at startup
and the panel can see exactly what is configurable without hunting through
the codebase.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised settings for the Healthcare AI Assistant."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- API server ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # --- Data + storage ---
    data_dir: Path = Field(default=Path("./data"))
    vector_store_dir: Path = Field(default=Path("./vector_store"))
    collection_name: str = "healthcare_docs"

    # --- Embeddings ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Chunking ---
    chunk_size: int = 500
    chunk_overlap: int = 50

    # --- Retrieval ---
    top_k: int = 4
    confidence_high_max_distance: float = 0.4
    confidence_medium_max_distance: float = 0.6

    # --- LLM (Ollama) ---
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    llm_temperature: float = 0.1
    llm_timeout_seconds: int = 60

    # --- UI ---
    ui_port: int = 8501
    api_base_url: str = "http://localhost:8000"

    # --- Derived helpers ---
    @property
    def data_dir_resolved(self) -> Path:
        return self.data_dir.expanduser().resolve()

    @property
    def vector_store_dir_resolved(self) -> Path:
        return self.vector_store_dir.expanduser().resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Cached so we do not re-parse ``.env`` on every request.
    """
    return Settings()
