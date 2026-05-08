"""Basic unit tests covering chunking, vector store, agent routing, and the
HTTP layer. The LLM and embedding model are stubbed (see conftest.py).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Ingestion / chunking


def test_ingest_indexes_synthetic_documents(tmp_path: Path):
    from app.config import get_settings
    from app.ingestion import ingest_directory
    from app.vectorstore import get_vector_store

    settings = get_settings()
    doc = settings.data_dir_resolved / "policy.md"
    doc.write_text(
        "# Heading\n\n"
        + "Para one. " * 80
        + "\n\n## Section\n\n"
        + "Para two. " * 80,
        encoding="utf-8",
    )

    stats = ingest_directory(reset=True)
    assert stats.files_processed == 1
    assert stats.chunks_indexed >= 1

    store = get_vector_store()
    assert store.count() == stats.chunks_indexed


def test_ingest_is_idempotent_on_unchanged_files():
    from app.config import get_settings
    from app.ingestion import ingest_directory
    from app.vectorstore import get_vector_store

    settings = get_settings()
    (settings.data_dir_resolved / "doc.md").write_text(
        "Some short content for chunking.", encoding="utf-8"
    )

    first = ingest_directory(reset=True)
    second = ingest_directory(reset=False)

    # Same content → identical chunk IDs → upsert leaves count unchanged.
    assert get_vector_store().count() == first.chunks_indexed == second.chunks_indexed


# ---------------------------------------------------------------------------
# Agent routing


def test_router_falls_back_to_heuristics_when_llm_unavailable():
    """Without the stub_llm fixture, chat() will fail because Ollama is
    unreachable in tests. The router should still classify reasonably."""
    from app.agent import classify_intent, Intent

    assert classify_intent("Can I book a cardiology appointment for Monday?") is Intent.APPOINTMENT
    assert classify_intent("What is the medication refill policy?") is Intent.KNOWLEDGE


def test_router_uses_llm_when_available(stub_llm):
    from app.agent import classify_intent, Intent

    stub_llm["intent_label"] = "appointment"
    assert classify_intent("hello") is Intent.APPOINTMENT

    stub_llm["intent_label"] = "out_of_scope"
    assert classify_intent("anything") is Intent.OUT_OF_SCOPE


def test_appointment_tool_returns_mock_slots():
    from app.tools import check_available_slots

    # Cardiology is open on Monday (weekday 0).
    next_monday = date.today()
    while next_monday.weekday() != 0:
        from datetime import timedelta
        next_monday = next_monday + timedelta(days=1)

    result = check_available_slots("cardiology", next_monday)
    assert result.available is True
    assert result.morning_slots, "expected morning slots for cardiology Monday"


def test_appointment_tool_handles_unknown_department():
    from app.tools import check_available_slots

    result = check_available_slots("astrology", "tomorrow")
    assert result.available is False
    assert "department" in (result.reason or "").lower()


# ---------------------------------------------------------------------------
# RAG end-to-end (with stubbed LLM)


def test_rag_returns_refusal_when_store_empty(stub_llm):
    from app.rag import answer_question

    result = answer_question("anything")
    assert result.confidence == "low"
    assert "knowledge base is empty" in result.answer.lower() or \
           "could not find" in result.answer.lower()


def test_rag_runs_full_loop_with_stubbed_llm(stub_llm):
    from app.config import get_settings
    from app.ingestion import ingest_directory
    from app.rag import answer_question

    settings = get_settings()
    (settings.data_dir_resolved / "telehealth.md").write_text(
        "## Telehealth refills\n\n"
        "Patients may request a medication refill during a telehealth visit "
        "if the medication has previously been prescribed.",
        encoding="utf-8",
    )
    ingest_directory(reset=True)

    stub_llm["answer"] = "Yes, telehealth refills are allowed for previously prescribed medications."
    result = answer_question("Can I refill medication via telehealth?")

    assert result.used_llm is True
    assert "telehealth" in result.answer.lower()
    # The disclaimer is appended.
    assert "not a substitute" in result.answer.lower()


# ---------------------------------------------------------------------------
# HTTP layer


def test_health_endpoint(stub_llm):
    from app.main import app

    with TestClient(app) as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"ok", "degraded"}
    assert "vector_store" in body
    assert "llm" in body


def test_ask_endpoint_validates_input(stub_llm):
    from app.main import app

    with TestClient(app) as client:
        resp = client.post("/ask", json={"question": ""})

    assert resp.status_code == 422  # Pydantic validation


def test_ask_endpoint_returns_structured_response(stub_llm):
    from app.config import get_settings
    from app.ingestion import ingest_directory
    from app.main import app

    settings = get_settings()
    (settings.data_dir_resolved / "refunds.md").write_text(
        "## Cancellations\n\nLate cancellations within 24 hours of the "
        "appointment are charged a flat fee of fifty rupees.",
        encoding="utf-8",
    )
    ingest_directory(reset=True)

    stub_llm["intent_label"] = "knowledge"
    stub_llm["answer"] = "A late cancellation within 24 hours is charged fifty rupees."

    with TestClient(app) as client:
        resp = client.post(
            "/ask",
            json={"question": "What is the cancellation fee?"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "knowledge"
    assert body["confidence"] in {"high", "medium", "low"}
    assert "fifty" in body["answer"].lower() or "cancellation" in body["answer"].lower()
