"""ChromaDB persistent vector store wrapper.

We use Chroma's PersistentClient so the index survives across process
restarts. Embeddings are computed on our side (in ``embeddings.py``)
rather than letting Chroma do it, so we keep tight control over the
embedding model and can swap it without re-creating the collection
from Chroma's perspective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.logger import get_logger, log_event

log = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A single chunk returned from a similarity search."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float

    @property
    def similarity(self) -> float:
        """Convenience: cosine similarity from cosine distance.

        Chroma returns distance in ``[0, 2]`` where 0 means identical.
        For normalised embeddings, ``similarity = 1 - distance``.
        """
        return 1.0 - self.distance


class VectorStore:
    """Thin wrapper around a Chroma collection."""

    def __init__(self) -> None:
        settings = get_settings()
        settings.vector_store_dir_resolved.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(settings.vector_store_dir_resolved),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        log_event(
            log,
            "vectorstore.ready",
            path=str(settings.vector_store_dir_resolved),
            collection=settings.collection_name,
            count=self._collection.count(),
        )

    # ------------------------------------------------------------------ writes

    def upsert(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        """Upsert a batch of chunks. Idempotent on chunk_id."""
        if not ids:
            return
        # Chroma's upsert accepts list-typed args.
        self._collection.upsert(
            ids=list(ids),
            documents=list(documents),
            embeddings=[list(v) for v in embeddings],
            metadatas=[dict(m) for m in metadatas],
        )
        log_event(log, "vectorstore.upsert", count=len(ids))

    def reset(self) -> None:
        """Drop and recreate the collection. Used by the /ingest endpoint
        when the caller passes ``reset=True``."""
        settings = get_settings()
        try:
            self._client.delete_collection(settings.collection_name)
        except Exception:  # noqa: BLE001 — Chroma raises a generic error when missing
            pass
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        log_event(log, "vectorstore.reset", collection=settings.collection_name)

    # ------------------------------------------------------------------- reads

    def count(self) -> int:
        return self._collection.count()

    def query(
        self,
        query_embedding: Sequence[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if self._collection.count() == 0:
            return []

        result = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=top_k,
        )

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        for chunk_id, text, meta, dist in zip(
            ids, documents, metadatas, distances, strict=False
        ):
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text or "",
                    metadata=dict(meta or {}),
                    distance=float(dist),
                )
            )
        return chunks


# A module-level singleton keeps the connection warm between requests.
_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def reset_vector_store_singleton() -> None:
    """Force a re-init on next call. Used after a full reset."""
    global _store
    _store = None
