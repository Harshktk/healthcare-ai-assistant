"""Lightweight agent router.

Classifies the user's question into one of three intents and dispatches
to the right handler:

    * ``knowledge``    → run the RAG pipeline over the document store.
    * ``appointment``  → call the mock ``check_available_slots`` tool.
    * ``out_of_scope`` → return a polite refusal.

We deliberately keep this simple. A small LLM call decides the intent;
if the LLM is unreachable we fall back to keyword heuristics so the demo
still works. Easy to explain in the panel and easy to extend.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.llm import LLMError, chat
from app.logger import get_logger, log_event
from app.prompts import ROUTER_SYSTEM_PROMPT, build_router_user_prompt
from app.rag import RAGAnswer, Source, answer_question
from app.tools import check_available_slots, extract_arguments, format_slot_response

log = get_logger(__name__)


class Intent(str, Enum):
    KNOWLEDGE = "knowledge"
    APPOINTMENT = "appointment"
    OUT_OF_SCOPE = "out_of_scope"
    GREETING = "greeting"


@dataclass
class AgentResult:
    answer: str
    sources: list[Source]
    confidence: str
    intent: Intent
    used_llm: bool
    tool_output: dict[str, Any] | None = None

    def as_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": [s.as_dict() for s in self.sources],
            "confidence": self.confidence,
            "intent": self.intent.value,
            "used_llm": self.used_llm,
            "tool_output": self.tool_output,
        }


# ---------------------------------------------------------------------------
# Intent classification

_APPOINTMENT_KEYWORDS = (
    "appointment", "book", "booking", "schedule a ", "scheduling a ",
    "available slot", "availability", "open slot", "free slot",
    "reschedule", "consultation slot",
)
_KNOWLEDGE_KEYWORDS = (
    "policy", "refill", "telehealth", "insurance", "hipaa",
    "discharge", "privacy", "covered", "rule", "guideline",
    "what is", "explain",
)


def _heuristic_intent(question: str) -> Intent:
    q = question.lower()
    appointment_signal = any(k in q for k in _APPOINTMENT_KEYWORDS)
    knowledge_signal = any(k in q for k in _KNOWLEDGE_KEYWORDS)

    if appointment_signal and not knowledge_signal:
        return Intent.APPOINTMENT
    if appointment_signal and knowledge_signal:
        # "What is the policy on booking?" → knowledge wins.
        return Intent.KNOWLEDGE
    if knowledge_signal:
        return Intent.KNOWLEDGE
    # Default: assume the user wants to consult the docs.
    return Intent.KNOWLEDGE


def classify_intent(question: str) -> Intent:
    """Strict heuristic: appointment only on explicit booking phrasing.
    Everything else → knowledge → RAG. Skips the LLM router for speed
    and reliability on small models."""
    q = question.lower()

    # Greeting: very short and clearly conversational.
    greetings = ("hi", "hello", "hey", "what can you", "help me", "who are you")
    if len(q.strip()) < 30 and any(g in q for g in greetings):
        return Intent.GREETING

    # Appointment: must contain a strong booking verb.
    strong_appointment_phrases = (
        "book", "booking", "schedule a ", "scheduling a ",
        "available slot", "any slot", "free slot", "open slot",
        "reschedule", "cancel my appointment",
    )
    if any(p in q for p in strong_appointment_phrases):
        return Intent.APPOINTMENT

    # Default — let RAG handle it. Refusal sentinel covers out-of-scope.
    return Intent.KNOWLEDGE


# ---------------------------------------------------------------------------
# Dispatch

_OUT_OF_SCOPE_REPLY = (
    "I am a healthcare clinic assistant. I can answer questions about clinic "
    "policies, telehealth, refills, insurance, privacy, and patient instructions, "
    "or check mock appointment availability. Could you rephrase your question "
    "to fit one of those areas?"
)


def _handle_appointment(question: str) -> AgentResult:
    department, parsed_date = extract_arguments(question)
    slots = check_available_slots(department, parsed_date)
    return AgentResult(
        answer=format_slot_response(slots),
        sources=[],
        confidence="high" if slots.available else "medium",
        intent=Intent.APPOINTMENT,
        used_llm=False,
        tool_output=slots.as_dict(),
    )


def _handle_knowledge(question: str) -> AgentResult:
    rag: RAGAnswer = answer_question(question)
    return AgentResult(
        answer=rag.answer,
        sources=rag.sources,
        confidence=rag.confidence,
        intent=Intent.KNOWLEDGE,
        used_llm=rag.used_llm,
    )


def _handle_out_of_scope() -> AgentResult:
    return AgentResult(
        answer=_OUT_OF_SCOPE_REPLY,
        sources=[],
        confidence="high",
        intent=Intent.OUT_OF_SCOPE,
        used_llm=False,
    )

_GREETING_REPLY = (
    "Hi! I'm a healthcare clinic assistant. I can answer questions about clinic "
    "policies, telehealth, medication refills, insurance eligibility, privacy "
    "guidelines, and discharge instructions. I can also check mock appointment "
    "availability — try asking 'Can I book a cardiology appointment for Monday?'"
)


def _handle_greeting() -> AgentResult:
    return AgentResult(
        answer=_GREETING_REPLY,
        sources=[],
        confidence="high",
        intent=Intent.GREETING,
        used_llm=False,
    )

def run(question: str) -> AgentResult:
    """Top-level entry point used by the API layer."""
    intent = classify_intent(question)
    log_event(log, "agent.routed", intent=intent.value)

    if intent is Intent.APPOINTMENT:
        return _handle_appointment(question)
    if intent is Intent.OUT_OF_SCOPE:
        return _handle_out_of_scope()
    if intent is Intent.GREETING:               
        return _handle_greeting()
    return _handle_knowledge(question)
