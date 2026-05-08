"""Document ingestion pipeline.

Reads files from the data directory, splits them into chunks, embeds
them, and upserts them into the vector store. Idempotent: re-running on
the same files produces no duplicates because chunk IDs are derived
from a content hash.

Supported file types: ``.md``, ``.txt``, ``.pdf``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.embeddings import embed_texts
from app.logger import get_logger, log_event
from app.vectorstore import get_vector_store

log = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


# ---------------------------------------------------------------------------
# Data classes


@dataclass
class IngestionStats:
    files_processed: int = 0
    files_skipped: int = 0
    chunks_indexed: int = 0
    documents_indexed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "files_processed": self.files_processed,
            "files_skipped": self.files_skipped,
            "chunks_indexed": self.chunks_indexed,
            "documents_indexed": self.documents_indexed,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# File loading


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    # Lazy import so installations without pypdf still load the module.
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001
            log.warning("pdf.extract_failed file=%s error=%s", path.name, exc)
    return "\n\n".join(pages)


def _load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".md", ".txt"):
        return _read_text_file(path)
    if suffix == ".pdf":
        return _read_pdf(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def _iter_supported_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


# ---------------------------------------------------------------------------
# Chunking


def _build_splitter() -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # Prefer splitting on markdown / paragraph boundaries first.
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def _chunk_id(document: str, chunk_index: int, text: str) -> str:
    """Deterministic ID from document name + chunk index + content hash.

    Including the hash means edits to the document produce a new ID for
    the changed chunks, so an upsert effectively replaces them.
    """
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    safe_doc = document.replace(" ", "_")
    return f"{safe_doc}::{chunk_index:04d}::{digest}"


# ---------------------------------------------------------------------------
# Public API


def ingest_directory(reset: bool = False) -> IngestionStats:
    """Ingest every supported file inside the configured ``data_dir``.

    Args:
        reset: If True, drop the collection before ingesting. Useful when
            documents have been deleted or chunking parameters changed.

    Returns:
        IngestionStats summarising the run.
    """
    settings = get_settings()
    store = get_vector_store()

    if reset:
        store.reset()

    splitter = _build_splitter()
    stats = IngestionStats()

    data_dir = settings.data_dir_resolved
    log_event(log, "ingestion.started", data_dir=str(data_dir), reset=reset)

    files = list(_iter_supported_files(data_dir))
    if not files:
        log_event(log, "ingestion.no_files_found", data_dir=str(data_dir))
        return stats

    for file_path in files:
        try:
            text = _load_document(file_path)
        except Exception as exc:  # noqa: BLE001
            stats.files_skipped += 1
            stats.errors.append(f"{file_path.name}: {exc}")
            log.exception("ingestion.load_failed file=%s", file_path.name)
            continue

        if not text.strip():
            stats.files_skipped += 1
            stats.errors.append(f"{file_path.name}: empty after extraction")
            continue

        chunks = splitter.split_text(text)
        if not chunks:
            stats.files_skipped += 1
            continue

        # Compute relative path for nicer citation labels.
        try:
            rel_path = file_path.relative_to(data_dir)
        except ValueError:
            rel_path = Path(file_path.name)

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for idx, chunk_text in enumerate(chunks):
            ids.append(_chunk_id(file_path.name, idx, chunk_text))
            documents.append(chunk_text)
            metadatas.append(
                {
                    "document": file_path.name,
                    "source_path": str(rel_path).replace("\\", "/"),
                    "chunk_index": idx,
                }
            )

        embeddings = embed_texts(documents)
        store.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        stats.files_processed += 1
        stats.chunks_indexed += len(chunks)
        stats.documents_indexed.append(file_path.name)
        log_event(
            log,
            "ingestion.file_indexed",
            file=file_path.name,
            chunks=len(chunks),
        )

    log_event(
        log,
        "ingestion.completed",
        files=stats.files_processed,
        skipped=stats.files_skipped,
        chunks=stats.chunks_indexed,
    )
    return stats
