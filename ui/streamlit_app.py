"""Streamlit chat UI.

Run locally:
    streamlit run ui/streamlit_app.py

The UI talks to the FastAPI backend via plain HTTP, so the two
processes can live in separate containers (see docker-compose.yml).
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TIMEOUT_SECONDS = 90

st.set_page_config(
    page_title="Healthcare AI Assistant",
    page_icon=":hospital:",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar


with st.sidebar:
    st.title("Healthcare AI Assistant")
    st.caption("RAG over clinic policy and patient instruction documents.")

    st.markdown("### Backend")
    st.code(API_BASE_URL, language="text")

    if st.button("Check health", use_container_width=True):
        try:
            resp = httpx.get(f"{API_BASE_URL}/health", timeout=10).json()
            st.json(resp)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Health check failed: {exc}")

    st.markdown("### Knowledge base")
    if st.button("Re-ingest documents", use_container_width=True):
        with st.spinner("Ingesting…"):
            try:
                resp = httpx.post(
                    f"{API_BASE_URL}/ingest",
                    json={"reset": True},
                    timeout=300,
                ).json()
                st.success(
                    f"Indexed {resp.get('chunks_indexed', 0)} chunks across "
                    f"{resp.get('files_processed', 0)} files."
                )
                if resp.get("errors"):
                    st.warning(resp["errors"])
            except Exception as exc:  # noqa: BLE001
                st.error(f"Ingest failed: {exc}")

    st.markdown("---")
    st.markdown("### Try asking")
    st.markdown(
        "- Can a patient request a medication refill through telehealth?\n"
        "- What happens if I cancel an appointment two hours before?\n"
        "- Do I need a referral to see a specialist?\n"
        "- Can I book a cardiology appointment for Monday?\n"
        "- When should I go to the emergency department after discharge?"
    )

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Chat state


if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# Helpers


_CONFIDENCE_BADGE = {
    "high": ":green[**high**]",
    "medium": ":orange[**medium**]",
    "low": ":red[**low**]",
}


def _render_assistant_payload(payload: dict) -> None:
    confidence = payload.get("confidence", "low")
    intent = payload.get("intent", "knowledge")
    used_llm = payload.get("used_llm", False)

    st.markdown(payload.get("answer", ""))

    meta_cols = st.columns(3)
    meta_cols[0].markdown(f"Confidence: {_CONFIDENCE_BADGE.get(confidence, confidence)}")
    meta_cols[1].markdown(f"Intent: `{intent}`")
    meta_cols[2].markdown(f"LLM used: `{used_llm}`")

    sources = payload.get("sources") or []
    if sources:
        with st.expander(f"Sources ({len(sources)})", expanded=False):
            for i, src in enumerate(sources, start=1):
                st.markdown(
                    f"**{i}. `{src.get('document', 'unknown')}`**  "
                    f"_(similarity {src.get('similarity', 0):.3f})_"
                )
                st.markdown(f"> {src.get('chunk', '')[:1000]}")

    if payload.get("tool_output"):
        with st.expander("Tool output (raw)", expanded=False):
            st.json(payload["tool_output"])


# ---------------------------------------------------------------------------
# Render history


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and isinstance(msg.get("payload"), dict):
            _render_assistant_payload(msg["payload"])
        else:
            st.markdown(msg.get("content", ""))


# ---------------------------------------------------------------------------
# Input loop


prompt = st.chat_input("Ask a healthcare or clinic-policy question…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with placeholder.container():
            with st.spinner("Thinking…"):
                try:
                    resp = httpx.post(
                        f"{API_BASE_URL}/ask",
                        json={"question": prompt},
                        timeout=TIMEOUT_SECONDS,
                    )
                    resp.raise_for_status()
                    payload = resp.json()
                except httpx.HTTPError as exc:
                    error_msg = f"Request failed: {exc}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )
                    st.stop()

        placeholder.empty()
        _render_assistant_payload(payload)
        st.session_state.messages.append({"role": "assistant", "payload": payload})
