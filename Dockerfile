# syntax=docker/dockerfile:1.6
#
# Single image used for both the FastAPI service and the Streamlit UI.
# docker-compose picks the right entrypoint per service.
#
# Build:  docker build -t healthcare-ai-assistant .
# Run:    docker run -p 8000:8000 healthcare-ai-assistant

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/hf \
    TRANSFORMERS_OFFLINE=0

# System packages needed by sentence-transformers (libgomp) and pypdf.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Dependency layer ---
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# --- Application layer ---
COPY app ./app
COPY ui ./ui
COPY data ./data
COPY eval ./eval
COPY .env.example ./.env.example

# Create writable directories used at runtime.
RUN mkdir -p /app/vector_store /app/.cache/hf

EXPOSE 8000 8501

# Default command runs the API. docker-compose overrides this for the UI.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
