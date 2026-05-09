# Architecture

This document describes the design of the Healthcare AI Assistant in two
layers: a **High-Level Design (HLD)** for orientation, and a **Low-Level
Design (LLD)** that drills into module-level contracts, data flow, and
key algorithms. It is intended to be read alongside `README.md` (which
focuses on setup and usage).

---

## Table of contents

1. [System overview](#1-system-overview)
2. [High-Level Design](#2-high-level-design-hld)
   1. [Goals and non-goals](#21-goals-and-non-goals)
   2. [System context diagram](#22-system-context-diagram)
   3. [Component diagram](#23-component-diagram)
   4. [Deployment topology](#24-deployment-topology)
   5. [Major data flows](#25-major-data-flows)
   6. [Cross-cutting concerns](#26-cross-cutting-concerns)
3. [Low-Level Design](#3-low-level-design-lld)
   1. [Module map](#31-module-map)
   2. [`config` and `logger`](#32-config-and-logger)
   3. [`embeddings`](#33-embeddings)
   4. [`vectorstore`](#34-vectorstore)
   5. [`ingestion`](#35-ingestion)
   6. [`prompts`](#36-prompts)
   7. [`llm`](#37-llm)
   8. [`rag`](#38-rag)
   9. [`tools`](#39-tools)
   10. [`agent`](#310-agent)
   11. [`schemas` and `main`](#311-schemas-and-main)
   12. [Streamlit UI](#312-streamlit-ui)
4. [Sequence diagrams](#4-sequence-diagrams)
5. [Data model](#5-data-model)
6. [Trade-offs and design decisions](#6-trade-offs-and-design-decisions)
7. [Production hardening checklist](#7-production-hardening-checklist)

---

## 1. System overview

The Healthcare AI Assistant is a **Retrieval-Augmented Generation (RAG)**
prototype that answers questions over a curated set of clinic
documents. It:

1. Ingests `.md`, `.txt`, `.pdf`, `.csv`, `.json`, and `.xml` documents.
2. Stores chunked sentence-transformer embeddings in a persistent
   ChromaDB vector store.
3. Routes each user question to one of four intents
   (`knowledge`, `appointment`, `out_of_scope`, `greeting`).
4. For knowledge questions, retrieves grounded context and asks a local
   LLM (Ollama + Mistral / Llama 3.2) to compose a citation-bearing
   answer; refuses if no chunk passes a similarity threshold.
5. For appointment questions, calls a mock scheduling tool.
6. Exposes everything through a FastAPI service with a Streamlit
   chat UI.

---

## 2. High-Level Design (HLD)

### 2.1 Goals and non-goals

**Goals**

- Demonstrate a complete, runnable RAG pipeline.
- Produce **grounded** answers with **explicit source citations**.
- **Refuse cleanly** when the documents do not contain the answer.
- Show a basic **agentic workflow** (router + mock tool).
- Stay **portable** (Docker compose), **private** (local LLM), and
  **extensible** (clean module boundaries).

**Non-goals (for this prototype)**

- Authentication / authorization.
- Real PHI handling, audit logging, encryption at rest.
- Multi-tenant document isolation.
- Conversational memory across turns.
- Streaming responses.
- Hybrid retrieval (BM25 + dense) or cross-encoder reranking.

### 2.2 System context diagram

```
                  +-----------------------+
                  |        User           |
                  | (browser / curl /     |
                  |  Postman)             |
                  +-----------+-----------+
                              |
                  HTTP        |
                              v
   +-------------+    +---------------+    +----------------+
   |  Streamlit  |--->|   FastAPI     |--->|    Ollama      |
   |     UI      |    |   backend     |    |  (local LLM)   |
   +-------------+    +-------+-------+    +----------------+
                              |
                              v
                      +---------------+
                      |  ChromaDB     |
                      |  (persistent) |
                      +---------------+
                              ^
                              |
                      +---------------+
                      |  data/        |
                      |  *.md, .pdf,  |
                      |  .csv, .json, |
                      |  .xml         |
                      +---------------+
```

### 2.3 Component diagram

```
+------------------------------ FastAPI (app/main.py) -----------------------+
|                                                                            |
|  /health    /ingest    /ask                                                |
|     |          |         |                                                 |
|     |          v         v                                                 |
|     |    +----------+   +-------------+    +--------------+                |
|     |    | Ingest   |   | Agent       |--->| Tools        |                |
|     |    | pipeline |   | router      |    | (mock slots) |                |
|     |    +----+-----+   +------+------+    +--------------+                |
|     |         |                |                                            |
|     |         v                v                                            |
|     |   +-----------+    +-----+-----+                                      |
|     +-->| LLM       |<---| RAG       |                                      |
|         | client    |    | pipeline  |                                      |
|         +-----+-----+    +-----+-----+                                      |
|               |                |                                            |
|               v                v                                            |
|         +-----------+    +-----------+                                      |
|         | Ollama    |    | Embeddings|                                      |
|         | runtime   |    | (Sentence |                                      |
|         +-----------+    | Transf.)  |                                      |
|                          +-----+-----+                                      |
|                                |                                            |
|                                v                                            |
|                          +-----------+                                      |
|                          | ChromaDB  |                                      |
|                          | persistent|                                      |
|                          +-----------+                                      |
+----------------------------------------------------------------------------+
```

### 2.4 Deployment topology

Three Docker services orchestrated by `docker-compose.yml`:

| Service  | Image / build              | Purpose                          | Port  |
| -------- | -------------------------- | -------------------------------- | ----- |
| `ollama` | `ollama/ollama:latest`     | Local LLM runtime                | 11434 |
| `api`    | `healthcare-ai-assistant`  | FastAPI + RAG + agent            | 8000  |
| `ui`     | `healthcare-ai-assistant`  | Streamlit chat frontend          | 8501  |

Persistent volumes:

- `ollama-models` — pulled LLMs (Mistral, Llama, …)
- `vector-store` — ChromaDB SQLite + HNSW files
- `hf-cache` — Hugging Face cache (sentence-transformers weights)
- `./data` (bind-mount) — ingestable documents from the host

Service dependency graph: `ui → api → ollama`. Health checks on
`ollama` and `api` ensure ordered startup.

### 2.5 Major data flows

**Ingestion (`POST /ingest`)**

```
client → /ingest (multipart files, reset?)
       → save uploads to data/uploads/
       → ingest_directory()
            for each supported file:
              load → chunk → embed → upsert (Chroma)
       → IngestResponse {files_processed, chunks_indexed, errors}
```

**Question answering (`POST /ask`)**

```
client → /ask {question}
       → agent.classify_intent()    [LLM call, with heuristic fallback]
            ├── knowledge   → rag.answer_question()
            │                   embed → retrieve top-k → confidence label
            │                   if low: refuse
            │                   else: LLM call with grounded prompt
            │                   attach citations from retrieval metadata
            ├── appointment → tools.check_available_slots()
            ├── out_of_scope→ static refusal
            └── greeting    → static help message
       → AskResponse {answer, sources[], confidence, intent, used_llm}
```

### 2.6 Cross-cutting concerns

| Concern         | Mechanism                                                     |
| --------------- | ------------------------------------------------------------- |
| Configuration   | `pydantic-settings`, `.env` driven, no hardcoded secrets      |
| Logging         | stdlib `logging` with `log_event()` for structured lines      |
| Error handling  | Global FastAPI exception handler + `LLMError` taxonomy        |
| Retries         | `tenacity` exponential backoff on transient `httpx` errors    |
| Timeouts        | `LLM_TIMEOUT_SECONDS` config, default 60s                     |
| Determinism     | Content-hash chunk IDs → idempotent ingestion                 |
| Safety          | RAG system prompt refuses diagnosis / dosage advice           |
| CORS            | Open in dev (`*`) — would tighten in production               |

---

## 3. Low-Level Design (LLD)

### 3.1 Module map

```
app/
├── __init__.py        version constant
├── main.py            FastAPI app, lifespan, endpoints
├── config.py          Settings (pydantic-settings)
├── logger.py          Logging setup + log_event helper
├── embeddings.py      sentence-transformers wrapper
├── vectorstore.py     ChromaDB wrapper, RetrievedChunk dataclass
├── ingestion.py       Loaders, splitter, ingest_directory()
├── prompts.py         RAG_SYSTEM_PROMPT, ROUTER_SYSTEM_PROMPT, builders
├── llm.py             Ollama client, chat(), health_check(), retry
├── rag.py             RAGAnswer, Source, answer_question()
├── tools.py           Mock check_available_slots(), arg extraction
├── agent.py           Intent enum, classify_intent(), run()
└── schemas.py         Pydantic request/response models
```

### 3.2 `config` and `logger`

**`config.Settings`** (pydantic-settings)

| Field                            | Default                                          | Notes                                            |
| -------------------------------- | ------------------------------------------------ | ------------------------------------------------ |
| `api_host`, `api_port`           | `0.0.0.0`, `8000`                                |                                                  |
| `data_dir`, `vector_store_dir`   | `./data`, `./vector_store`                       | Resolved to absolute via property                |
| `embedding_model`                | `sentence-transformers/all-MiniLM-L6-v2`         | 384-dim, CPU-friendly                            |
| `chunk_size`, `chunk_overlap`    | `500`, `50`                                      | RecursiveCharacterTextSplitter                   |
| `top_k`                          | `4`                                              | Retrieved chunks per query                       |
| `confidence_high_max_distance`   | `0.5`                                            | Cosine distance threshold for HIGH               |
| `confidence_medium_max_distance` | `0.75`                                           | Cosine distance threshold for MEDIUM             |
| `ollama_host`, `llm_model`       | `http://localhost:11434`, `llama3.2:3b`          | Override per env (e.g. `mistral:latest`)         |
| `llm_temperature`                | `0.1`                                            | Low temp for deterministic answers               |
| `llm_timeout_seconds`            | `60`                                             | Increase for verbose enumerations                |

`get_settings()` is `lru_cache`-d so `.env` is parsed once.

**`logger.get_logger(name)`** configures a single stdout handler with a
fixed format. `log_event(logger, "event.name", k=v, ...)` emits a
structured one-liner with key=value pairs — easy to grep, easy to ship
to a log aggregator later.

### 3.3 `embeddings`

```python
def embed_texts(texts: Sequence[str]) -> list[list[float]]
def embed_query(query: str) -> list[float]
def embedding_dimension() -> int
```

- Uses `sentence-transformers` with `normalize_embeddings=True` so
  cosine similarity equals dot product and Chroma's `cosine` distance
  behaves predictably.
- The model is loaded **lazily** (`_load_model` with `lru_cache`) — keeps
  module import cheap and lets tests stub the functions without
  installing PyTorch.

### 3.4 `vectorstore`

```python
@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict
    distance: float

    @property
    def similarity(self) -> float  # 1.0 - distance

class VectorStore:
    def upsert(ids, documents, embeddings, metadatas) -> None
    def reset() -> None
    def count() -> int
    def query(query_embedding, top_k) -> list[RetrievedChunk]
```

- Wraps `chromadb.PersistentClient` so the index survives restarts.
- The collection is created with `{"hnsw:space": "cosine"}` to match the
  normalised embeddings.
- We compute embeddings on our side (not via Chroma's embedding
  function) so the LLM/embedding stack is portable.
- Exposed as a module-level singleton via `get_vector_store()`. Tests
  call `reset_vector_store_singleton()` between runs.

### 3.5 `ingestion`

**Pipeline**

```
data_dir.rglob(*) → filter SUPPORTED_EXTENSIONS
                 → _load_document() per format
                 → RecursiveCharacterTextSplitter
                 → embed_texts()
                 → store.upsert(ids, docs, embeddings, metadatas)
```

**Loaders**

| Suffix          | Function          | Notes                                           |
| --------------- | ----------------- | ----------------------------------------------- |
| `.md`, `.txt`   | `_read_text_file` | UTF-8, replace errors                           |
| `.pdf`          | `_read_pdf`       | `pypdf`, per-page extract, log on failure       |
| `.csv`          | `_read_csv`       | Each row → `col1 \| col2 \| …`                  |
| `.json`         | `_read_json`      | Pretty-printed (indent=2) for chunkable text    |
| `.xml`          | `_read_xml`       | Walks tree, emits `tag: text [attrs]` lines     |

**Chunking**

`RecursiveCharacterTextSplitter` with markdown-aware separators
(`\n## `, `\n### `, `\n\n`, `\n`, `. `, ` `, `""`). 500 chars / 50
overlap is tuned for short policy docs.

**Idempotency**

`_chunk_id(document, idx, text)` returns
`{doc}::{idx:04d}::{sha1[:10]}`. Identical content re-ingests as a
no-op upsert; edited content produces a different hash so the upsert
replaces the old chunk.

**Public API**

```python
def ingest_directory(reset: bool = False) -> IngestionStats
def ingest_single_file(file_path: Path) -> IngestionStats   # used by /ingest uploads
```

### 3.6 `prompts`

Two prompts, one place. Reviewers can read every instruction the LLM
sees by opening this single file.

- `RAG_SYSTEM_PROMPT` — six numbered rules: use only context, refuse
  with sentinel sentence, no invented values, no diagnosis/dosage,
  concise and professional, no model-emitted "Sources:" section.
- `ROUTER_SYSTEM_PROMPT` — classifies into one of four intents with
  in-context examples.
- `build_rag_user_prompt(question, context_blocks)` wraps each chunk
  with a `[Context N]` tag.
- `MEDICAL_DISCLAIMER` is appended in `rag.py`, never asked of the LLM
  (deterministic insurance).
- `REFUSAL_NO_CONTEXT` is the exact sentinel both the system prompt and
  the application use, so refusal detection by string match is reliable.

### 3.7 `llm`

```python
@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(...),
       retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)))
def chat(system_prompt: str, user_prompt: str,
         *, temperature=None, model=None) -> LLMResponse

def health_check() -> dict   # never raises; reports ok=False on failure
```

- Wraps the official `ollama` Python client.
- Retries only on transient transport errors (`tenacity`); 4xx-style
  validation errors fail fast with `LLMError`.
- `health_check` is best-effort and used by `GET /health`.

### 3.8 `rag`

```python
@dataclass
class Source:
    document: str
    chunk: str
    chunk_id: str
    similarity: float

@dataclass
class RAGAnswer:
    answer: str
    sources: list[Source]
    confidence: Literal["high", "medium", "low"]
    used_llm: bool

def answer_question(question: str) -> RAGAnswer
```

**Decision points**

1. Empty store → return a friendly "knowledge base is empty" message,
   no LLM call.
2. No chunks returned → refusal sentinel.
3. Top distance > `CONFIDENCE_MEDIUM_MAX_DISTANCE` → refusal **without
   LLM call** (cheapest path, prevents hallucination).
4. Otherwise → call LLM. If LLM throws, return retrieved chunks as
   sources but flag `used_llm=false` and a degraded message — the
   reviewer can still see what was retrieved.
5. If the LLM's answer starts with the refusal sentinel, drop the
   sources (not implying they were used).
6. Append the static medical disclaimer.

**Confidence ladder**

| Top distance         | Label    |
| -------------------- | -------- |
| `<= 0.5`             | high     |
| `<= 0.75`            | medium   |
| otherwise            | low      |

**Citation policy** — citations come from retrieval metadata, not the
LLM's text. This eliminates hallucinated sources.

### 3.9 `tools`

```python
@dataclass
class SlotAvailability:
    department: str
    date: str
    available: bool
    reason: str | None
    morning_slots: list[str]
    afternoon_slots: list[str]

def check_available_slots(department, date_value) -> SlotAvailability
def extract_arguments(question: str) -> tuple[str | None, date | None]
def format_slot_response(slots: SlotAvailability) -> str
```

- `_DEPARTMENT_SCHEDULES` is a small in-memory table — easy for the
  panel to inspect and modify live.
- `_DEPARTMENT_ALIASES` maps colloquial terms (`gp`, `kids`, `heart`) to
  canonical departments.
- `_parse_date` understands `today`, `tomorrow`, weekdays, ISO dates,
  and `dd/mm[/yyyy]`. Falls back to `today + 1` if nothing parses.

### 3.10 `agent`

```python
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
    tool_output: dict | None

def classify_intent(question: str) -> Intent
def run(question: str) -> AgentResult
```

- `classify_intent` calls the router prompt at `temperature=0.0`. If the
  LLM is unreachable, falls back to keyword heuristics
  (`_heuristic_intent`).
- The fallback parser is forgiving: it scans the LLM's tokens for the
  first known intent word, so an answer like `"appointment."` or
  `"appointment, because…"` is still parsed correctly.
- `run` dispatches: appointment → mock tool, out_of_scope/greeting →
  static reply, knowledge → RAG.

### 3.11 `schemas` and `main`

**Pydantic models** (`schemas.py`)

```python
class IngestRequest(BaseModel):
    reset: bool = False

class IngestResponse(BaseModel):
    files_processed: int
    files_skipped: int
    chunks_indexed: int
    documents_indexed: list[str]
    errors: list[str]

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

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    vector_store: dict
    llm: dict
```

**Endpoints** (`main.py`)

| Endpoint        | Purpose                                              |
| --------------- | ---------------------------------------------------- |
| `GET  /health`  | Vector store chunk count + Ollama reachability       |
| `POST /ingest`  | Optional multipart upload + re-ingest data folder    |
| `POST /ask`     | Run agent on a question, return grounded answer      |

A `lifespan` context warms the vector store on startup so the first
request is not penalised by cold initialisation. A global exception
handler converts unhandled exceptions into JSON 500s with a logged
traceback.

### 3.12 Streamlit UI

`ui/streamlit_app.py` is a single-file chat interface that talks to the
FastAPI backend over plain HTTP — the same surface a real client would
consume.

- **Sidebar (admin):** health check, "Re-ingest data folder (reset)",
  try-asking hints, clear conversation.
- **Inline above chat input:** collapsible "Attach a document" panel
  with multi-file upload + "Ingest now" (no reset, additive).
- **Per-message rendering:** confidence badge, intent tag, `used_llm`
  flag, expandable source citations with similarity scores, and an
  expandable "Tool output (raw)" panel for appointment results.

---

## 4. Sequence diagrams

### 4.1 Knowledge question (happy path)

```
User → UI: "What is the cancellation policy?"
UI   → API: POST /ask
API  → Agent: classify_intent()
Agent→ LLM: router prompt
LLM  → Agent: "knowledge"
Agent→ RAG: answer_question()
RAG  → Embeddings: embed_query()
RAG  → Chroma: query(top_k=4)
Chroma→RAG: 4 chunks
RAG  → RAG: confidence label = high
RAG  → LLM: system + grounded user prompt
LLM  → RAG: answer text
RAG  → API: RAGAnswer{answer, sources, confidence}
API  → UI: AskResponse
UI   → User: rendered answer + cited chunks
```

### 4.2 Appointment question (tool path)

```
User → UI: "Can I book a cardiology appointment for Monday?"
UI   → API: POST /ask
API  → Agent: classify_intent()
Agent→ LLM: router prompt → "appointment"
Agent→ Tools: extract_arguments → ("cardiology", next Monday)
Tools→ Agent: SlotAvailability{available=True, slots=[...]}
Agent→ API: AgentResult{intent=appointment, tool_output}
API  → UI: AskResponse
UI   → User: formatted slots + raw tool JSON
```

### 4.3 Refusal (no relevant context)

```
User → UI: "What is the recommended insulin dose?"
UI   → API: POST /ask
API  → Agent → RAG: answer_question()
RAG  → Chroma: query(top_k=4)
Chroma→RAG: 4 chunks (top_distance=0.82)
RAG  → RAG: confidence = low → refuse without LLM call
RAG  → API: RAGAnswer{"I could not find...", sources=[], used_llm=false}
API  → UI: AskResponse
UI   → User: refusal + empty sources
```

### 4.4 Ingestion with upload

```
User → UI: drop CSV + click Ingest now
UI   → API: POST /ingest (multipart, reset=false)
API  → FS: write file to data/uploads/
API  → Ingestion: ingest_directory()
Ingestion→FS: scan data/ recursively
loop per file:
    Ingestion → loader → text
    Ingestion → splitter → chunks
    Ingestion → Embeddings → vectors
    Ingestion → Chroma → upsert(ids, docs, embeddings, metas)
Ingestion→ API: IngestionStats
API  → UI: IngestResponse{chunks_indexed, files_processed, errors}
UI   → User: success toast
```

---

## 5. Data model

### 5.1 Chunk record (Chroma)

| Field        | Type         | Source                                         |
| ------------ | ------------ | ---------------------------------------------- |
| `id`         | `str`        | `{doc}::{chunk_idx:04d}::{sha1[:10]}`          |
| `document`   | `str`        | text content of the chunk                      |
| `embedding`  | `list[float]`| 384-dim sentence-transformers MiniLM           |
| `metadata`   | `dict`       | `{document, source_path, chunk_index}`         |

### 5.2 API response (`/ask`)

```json
{
  "answer": "…\n\n_This information is for general guidance only…_",
  "sources": [
    {
      "document": "telehealth_policy.md",
      "chunk": "Patients may request a medication refill during a telehealth visit if…",
      "chunk_id": "telehealth_policy.md::0002::3a1f2c9b08",
      "similarity": 0.7421
    }
  ],
  "confidence": "high",
  "intent": "knowledge",
  "used_llm": true,
  "tool_output": null
}
```

---

## 6. Trade-offs and design decisions

| Decision                                | Why                                                  | What we gave up                              |
| --------------------------------------- | ---------------------------------------------------- | -------------------------------------------- |
| Custom router over LangChain/CrewAI     | Easier to explain live, fewer moving parts           | Pre-built tool ecosystems                    |
| Citations from metadata, not LLM output | Eliminates hallucinated sources                      | LLM cannot reference chunks by tag in prose  |
| Confidence from retrieval distance      | Cheap, deterministic, refuses before LLM call        | Doesn't capture answer-side uncertainty      |
| ChromaDB embedded                       | One pip install, no extra server                     | Less production-credible than Qdrant/Weaviate |
| Local Ollama LLM                        | Privacy story for healthcare, no third-party data    | Slower than hosted APIs on CPU               |
| Per-doc upserts with content-hash IDs   | Idempotent re-ingestion                              | Doesn't track deletions automatically        |
| Single `/ingest` for upload + rebuild   | Simplest API surface, one mental model               | Slower than per-file delta ingestion at scale |

---

## 7. Production hardening checklist

A non-exhaustive list of what would change to take this from prototype
to production over real PHI:

- **Auth.** OAuth2/OIDC with role-based access; API keys for service
  callers; per-collection scopes.
- **Network.** mTLS between services, no open CORS, secrets via a
  vault.
- **Storage.** Encrypt the vector store at rest (LUKS / cloud KMS).
- **Audit.** Tamper-evident audit log of every `/ask` (request id,
  principal, retrieved chunk IDs, answer hash). Do **not** log chunk
  text or answer body — that creates new PHI artefacts.
- **PHI hygiene.** Pre-ingestion redaction (regex + NER) for stray
  identifiers; document-level access lists tracked alongside metadata.
- **Right-to-be-forgotten.** Maintain `document → chunk_ids` mapping so
  a deletion request purges all derived embeddings.
- **Retrieval quality.** Hybrid search (BM25 + dense) with reciprocal
  rank fusion; cross-encoder reranking on top 20 candidates.
- **Eval.** Move from keyword-based to LLM-as-judge or human review;
  track answer correctness over time.
- **Streaming.** Server-sent events on `/ask` so the UI shows tokens as
  they arrive.
- **Observability.** OpenTelemetry traces (the chain
  router → retrieve → generate is a textbook trace), Prometheus
  metrics for retrieval latency, LLM tokens, cache hits.
- **Rate limiting.** Per-principal quotas at the gateway.
- **Vector DB swap.** Replace `vectorstore.py` with a Qdrant or Weaviate
  client; the rest of the code is unaffected because the wrapper
  surface is small (`upsert`, `query`, `count`, `reset`).
