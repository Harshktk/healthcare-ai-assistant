"""Pydantic request/response models for the API.

Keeping them in one file makes the API contract easy to review and gives
FastAPI everything it needs to generate Swagger docs at ``/docs``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# /ingest


class IngestRequest(BaseModel):
    reset: bool = Field(
        default=False,
        description="If true, drop the existing collection before ingesting.",
    )


class IngestResponse(BaseModel):
    files_processed: int
    files_skipped: int
    chunks_indexed: int
    documents_indexed: list[str]
    errors: list[str]


# ---------------------------------------------------------------------------
# /ask


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class SourceModel(BaseModel):
    document: str
    chunk: str
    chunk_id: str
    similarity: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceModel]
    confidence: Literal["high", "medium", "low"]
    intent: Literal["knowledge", "appointment", "out_of_scope", "greeting"]
    used_llm: bool
    tool_output: dict | None = None


# ---------------------------------------------------------------------------
# /health


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    vector_store: dict
    llm: dict
