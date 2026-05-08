"""Retrieval-Augmented Generation pipeline.

Steps:
    1. Embed the user's question.
    2. Retrieve top-k chunks from the vector store.
    3. If no chunk clears the confidence floor, return the refusal directly
       without calling the LLM (saves time and prevents hallucination).
    4. Otherwise call the LLM with the system prompt + retrieved context.
    5. Compute a confidence label from the top retrieval distance.
    6. Return the answer with structured source citations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.config import get_settings
from app.embeddings import embed_query
from app.llm import LLMError, chat
from app.logger import get_logger, log_event
from app.prompts import (
    MEDICAL_DISCLAIMER,
    RAG_SYSTEM_PROMPT,
    REFUSAL_NO_CONTEXT,
    build_rag_user_prompt,
)
from app.vectorstore import RetrievedChunk, get_vector_store

log = get_logger(__name__)

ConfidenceLabel = Literal["high", "medium", "low"]


@dataclass
class Source:
    document: str
    chunk: str
    chunk_id: str
    similarity: float

    def as_dict(self) -> dict:
        return {
            "document": self.document,
            "chunk": self.chunk,
            "chunk_id": self.chunk_id,
            "similarity": round(self.similarity, 4),
        }


@dataclass
class RAGAnswer:
    answer: str
    sources: list[Source]
    confidence: ConfidenceLabel
    used_llm: bool

    def as_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": [s.as_dict() for s in self.sources],
            "confidence": self.confidence,
            "used_llm": self.used_llm,
        }


def _label_confidence(top_distance: float | None) -> ConfidenceLabel:
    """Map the best (smallest) distance to a confidence label."""
    settings = get_settings()
    if top_distance is None:
        return "low"
    if top_distance <= settings.confidence_high_max_distance:
        return "high"
    if top_distance <= settings.confidence_medium_max_distance:
        return "medium"
    return "low"


def _to_sources(chunks: list[RetrievedChunk]) -> list[Source]:
    sources: list[Source] = []
    for c in chunks:
        sources.append(
            Source(
                document=str(c.metadata.get("document", "unknown")),
                chunk=c.text,
                chunk_id=c.chunk_id,
                similarity=max(0.0, min(1.0, c.similarity)),
            )
        )
    return sources


def answer_question(question: str) -> RAGAnswer:
    """Run the full RAG pipeline for a single question."""
    settings = get_settings()
    store = get_vector_store()

    if not question or not question.strip():
        return RAGAnswer(
            answer="Please provide a non-empty question.",
            sources=[],
            confidence="low",
            used_llm=False,
        )

    if store.count() == 0:
        log_event(log, "rag.empty_store")
        return RAGAnswer(
            answer=(
                "The knowledge base is empty. Run POST /ingest (or the "
                "ingestion script) before asking questions."
            ),
            sources=[],
            confidence="low",
            used_llm=False,
        )

    # 1+2. Embed and retrieve.
    query_vec = embed_query(question)
    chunks = store.query(query_vec, top_k=settings.top_k)
    log_event(
        log,
        "rag.retrieved",
        top_k=settings.top_k,
        returned=len(chunks),
        top_distance=chunks[0].distance if chunks else None,
    )

    if not chunks:
        return RAGAnswer(
            answer=REFUSAL_NO_CONTEXT,
            sources=[],
            confidence="low",
            used_llm=False,
        )

    top_distance = chunks[0].distance
    confidence = _label_confidence(top_distance)

    # 3. Refuse early if no chunk is even moderately relevant.
    if confidence == "low":
        log_event(log, "rag.refused_low_similarity", top_distance=top_distance)
        return RAGAnswer(
            answer=REFUSAL_NO_CONTEXT,
            sources=[],  # do not return weak matches as "sources"
            confidence="low",
            used_llm=False,
        )

    # 4. Generate.
    context_blocks = [c.text for c in chunks]
    user_prompt = build_rag_user_prompt(question, context_blocks)

    try:
        llm_response = chat(
            system_prompt=RAG_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
    except LLMError as exc:
        log_event(log, "rag.llm_failed", error=str(exc))
        return RAGAnswer(
            answer=(
                "The language model is currently unreachable, so I cannot "
                "compose a grounded answer. Relevant passages were retrieved "
                "and are listed in sources for manual review."
            ),
            sources=_to_sources(chunks),
            confidence=confidence,
            used_llm=False,
        )

    raw_answer = llm_response.text or REFUSAL_NO_CONTEXT

    # 5. If the model itself refused, do not attach sources we'd be implying
    #    were used.
    if raw_answer.strip().lower().startswith(REFUSAL_NO_CONTEXT.lower()[:30]):
        return RAGAnswer(
            answer=REFUSAL_NO_CONTEXT,
            sources=[],
            confidence="low",
            used_llm=True,
        )

    # 6. Append the disclaimer once. Cheap insurance for the panel demo.
    final_answer = f"{raw_answer.strip()}\n\n_{MEDICAL_DISCLAIMER}_"

    return RAGAnswer(
        answer=final_answer,
        sources=_to_sources(chunks),
        confidence=confidence,
        used_llm=True,
    )
