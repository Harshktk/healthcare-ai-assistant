"""Ollama LLM client.

We talk to Ollama over its HTTP API via the official ``ollama`` Python
client. Wrapped here so we can swap providers (Groq, OpenAI, vLLM, …)
without touching the rest of the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import ollama
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings
from app.logger import get_logger, log_event

log = get_logger(__name__)


class LLMError(RuntimeError):
    """Raised when the LLM call fails after retries."""


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None


def _client() -> ollama.Client:
    settings = get_settings()
    return ollama.Client(
        host=settings.ollama_host,
        timeout=settings.llm_timeout_seconds,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    reraise=True,
)
def chat(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float | None = None,
    model: str | None = None,
) -> LLMResponse:
    """Call the configured LLM with a system + user message.

    Retries on transient network errors but not on validation errors.
    """
    settings = get_settings()
    chosen_model = model or settings.llm_model
    chosen_temp = settings.llm_temperature if temperature is None else temperature

    log_event(log, "llm.request", model=chosen_model, temperature=chosen_temp)

    try:
        response = _client().chat(
            model=chosen_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": chosen_temp},
        )
    except (httpx.HTTPError, ollama.ResponseError) as exc:
        log.exception("llm.request_failed model=%s", chosen_model)
        raise LLMError(
            f"LLM call failed (model={chosen_model}). "
            "Is Ollama running and the model pulled? "
            f"Underlying error: {exc}"
        ) from exc

    message = response.get("message") or {}
    text = (message.get("content") or "").strip()

    log_event(log, "llm.response", model=chosen_model, length=len(text))

    return LLMResponse(
        text=text,
        model=chosen_model,
        prompt_tokens=response.get("prompt_eval_count"),
        completion_tokens=response.get("eval_count"),
    )


def health_check() -> dict:
    """Return a small dict describing whether Ollama is reachable.

    Used by the ``/health`` endpoint. Never raises — failures are reported
    as ``ok=False``.
    """
    settings = get_settings()
    try:
        models = _client().list()
        names = [m.get("name") for m in models.get("models", []) if m.get("name")]
        return {
            "ok": True,
            "host": settings.ollama_host,
            "configured_model": settings.llm_model,
            "model_available": settings.llm_model in names,
            "available_models": names,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "host": settings.ollama_host,
            "configured_model": settings.llm_model,
            "error": str(exc),
        }
