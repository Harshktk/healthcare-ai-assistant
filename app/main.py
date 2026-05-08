"""FastAPI application entrypoint.

Exposes:
    POST /ingest  — re-ingest the data folder into the vector store.
    POST /ask     — ask a question; returns answer + citations + confidence.
    GET  /health  — lightweight liveness/readiness probe.

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.agent import run as run_agent
from app.config import get_settings
from app.ingestion import ingest_directory
from app.llm import health_check as llm_health
from app.logger import get_logger, log_event
from app.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SourceModel,
)
from app.vectorstore import get_vector_store

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    log_event(
        log,
        "app.startup",
        version=__version__,
        host=settings.api_host,
        port=settings.api_port,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
    )
    # Warm up the vector store handle so the first request is not penalised.
    get_vector_store()
    yield
    log_event(log, "app.shutdown")


app = FastAPI(
    title="Healthcare AI Assistant",
    description=(
        "RAG-based assistant for clinic policies, patient instructions, and "
        "general healthcare information. Built for the Mindbowser AI Engineer "
        "hackathon."
    ),
    version=__version__,
    lifespan=lifespan,
)

# Open CORS for local Streamlit + curl / Postman use. Tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Error handlers


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled.exception path=%s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error. See server logs for details.",
            "type": exc.__class__.__name__,
        },
    )


# ---------------------------------------------------------------------------
# Endpoints


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    """Quick probe describing the state of the vector store and the LLM."""
    store = get_vector_store()
    vs_info: dict[str, Any] = {
        "ok": True,
        "chunk_count": store.count(),
    }
    llm_info = llm_health()

    status = "ok" if vs_info["ok"] and llm_info.get("ok") else "degraded"
    return HealthResponse(
        status=status,  # type: ignore[arg-type]
        version=__version__,
        vector_store=vs_info,
        llm=llm_info,
    )


@app.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
async def ingest(req: IngestRequest) -> IngestResponse:
    """Re-ingest every supported file in the configured data directory."""
    try:
        stats = ingest_directory(reset=req.reset)
    except Exception as exc:  # noqa: BLE001
        log.exception("ingest.failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return IngestResponse(**stats.as_dict())


@app.post("/ask", response_model=AskResponse, tags=["qa"])
async def ask(req: AskRequest) -> AskResponse:
    """Run the agent on the user's question and return a grounded answer."""
    log_event(log, "ask.received", length=len(req.question))
    result = run_agent(req.question)
    return AskResponse(
        answer=result.answer,
        sources=[SourceModel(**s.as_dict()) for s in result.sources],
        confidence=result.confidence,  # type: ignore[arg-type]
        intent=result.intent.value,  # type: ignore[arg-type]
        used_llm=result.used_llm,
        tool_output=result.tool_output,
    )


# ---------------------------------------------------------------------------
# Convenience root


@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "service": "healthcare-ai-assistant",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }
