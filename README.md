# KSK Shrama Sahayak — RAG Chatbot for KBOCWWB

A production-grade **Retrieval-Augmented Generation (RAG)** chatbot built for the **Karnataka Building & Other Construction Workers Welfare Board (KBOCWWB)** and its **Karmika Seva Kendras (KSK)**. The chatbot — named **Shrama Sahayak** (ಶ್ರಮ ಸಹಾಯಕ, "helper of working people") — answers questions about welfare schemes, registration, renewal, and application status for construction workers in Karnataka, India.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Security](#security)
- [Testing](#testing)
- [Docker Deployment](#docker-deployment)
- [Project Structure](#project-structure)

---

## Overview

Construction workers registered under KBOCWWB can access welfare schemes such as financial assistance for education, marriage, medical treatment, housing, and more. This chatbot ingests Markdown documentation about these schemes, stores vector embeddings in Qdrant, and generates context-aware answers using the **Devstral 24B** LLM via Ollama.

**Key capabilities:**

- Answers general questions about schemes, eligibility, documents, and procedures
- Checks real-time application/registration/renewal status via the Karnataka government backend API
- Supports authenticated users with personalized responses using their actual data
- Handles e-card requests by returning a signal for the frontend to display the card
- Works in 7 languages: English, Kannada, Hindi, Tamil, Telugu, Malayalam, Marathi

**Tech stack:** FastAPI | Ollama (Devstral 24B) | Qdrant | SQLite (aiosqlite) | httpx | Python 3.10+

---

## Features

### Intent Classification (Two-Tier)

1. **Layer 1 — Keyword matching** (all users): Deterministic, zero-cost substring matching against curated English and Kannada keyword lists. Unicode NFC normalization handles invisible character differences across input methods.
2. **Layer 2 — LLM classification** (authenticated users only): For messages that don't match any keyword, a constrained LLM call (temperature=0, top_k=1, num_predict=10) classifies the intent as `ECARD`, `STATUS_CHECK`, or `GENERAL`.

### Auth-Aware Behavior

| User State | ECARD Intent | STATUS_CHECK Intent | GENERAL Intent |
|---|---|---|---|
| **Anonymous** | Returns `<<LOGIN_MODAL_REQUIRED>>` | Returns `<<LOGIN_MODAL_REQUIRED>>` | RAG pipeline (context only) |
| **Logged in** | Returns `ECARD` constant | LLM grounded on fetched user data | RAG pipeline + user data |

### Multilingual Support

Responses can be generated in any of the 7 supported languages. Kannada receives special handling with native-speaker-quality instructions, proper script usage (not transliteration), and culturally appropriate fallback messages.

### Streaming Responses

Server-Sent Events (SSE) endpoint for real-time streaming. Events:
- `chunk` — incremental text as it arrives from the LLM
- `done` — final event with full answer, threadId, and messageId
- `error` — timeout or generation failure with error code

### External API Integration

Fetches real-time data from the Karnataka government backend (`apikbocwwb.karnataka.gov.in`):
- **Schemes** — applied schemes, approval status, rejection reasons
- **Registration** — registration code, status, renewal status
- **Renewal date** — card/registration renewal date

All three are fetched in parallel via `asyncio.gather`. Per-scheme status lookups also run concurrently. Results are cached per-thread in SQLite.

### Production Security Hardening

- IP-based sliding-window rate limiting with memory exhaustion protection
- Security headers: HSTS, CSP, X-Frame-Options, Permissions-Policy, Referrer-Policy
- Input validation: Pydantic schemas, userId regex, UUID format checks
- Auth token leak prevention (no tracebacks logged for external API errors)
- Response size guards on all Ollama endpoints
- LLM answer length cap
- Streaming timeout protection
- Configurable proxy header trust for reverse proxy deployments

---

## Architecture

### Request Flow

```
Client
  │
  ├─ POST /api/chat/threads                    → Create thread (UUID)
  │
  ├─ POST /api/chat/threads/{id}/messages       → Non-streaming
  │   │
  │   ├─ Validate thread (SQLite)
  │   ├─ Save user message
  │   ├─ Fetch history (recent N messages)
  │   ├─ classify_and_prepare() [under thread lock]
  │   │   ├─ Layer 1: keyword match
  │   │   ├─ Layer 2: LLM classify (if auth + no keyword hit)
  │   │   ├─ If ECARD/STATUS_CHECK + anon → LOGIN_REQUIRED
  │   │   ├─ If ECARD + auth → return ECARD constant
  │   │   └─ If STATUS_CHECK/GENERAL + auth → fetch/cache user data
  │   ├─ answer()
  │   │   ├─ LOGIN_REQUIRED → "<<LOGIN_MODAL_REQUIRED>>"
  │   │   ├─ ECARD → "ECARD"
  │   │   ├─ STATUS_CHECK → LLM(status_prompt + user_data + history)
  │   │   └─ GENERAL → retrieve(query→embed→Qdrant) → LLM(prompt + context [+ user_data] + history)
  │   └─ Save assistant message, return response
  │
  ├─ POST /api/chat/threads/{id}/messages/stream → SSE streaming (same flow, yields chunks)
  │
  ├─ GET /api/chat/threads/{id}/messages        → Paginated message history
  │
  ├─ GET /health                                → Liveness probe
  └─ GET /ready                                 → Readiness probe (DB + Qdrant + Ollama)
```

### Ingestion Flow

```
data/ksk.md
  → app/chunker.py (split by # and ## headers via langchain)
  → app/ollama_client.py (embed each chunk with nomic-embed-text)
  → Qdrant upsert (768-dim cosine vectors)
```

### Key Modules

| Module | Responsibility |
|---|---|
| `app/main.py` | FastAPI app, middleware stack, endpoints, lifespan management |
| `app/rag.py` | Intent classification, system prompts, RAG retrieval, answer generation |
| `app/ollama_client.py` | Async Ollama HTTP client (chat, stream, classify, generate, embed) |
| `app/database.py` | SQLite schema, CRUD, user data cache, retention cleanup |
| `app/external_api.py` | Karnataka govt API client (schemes, registration, renewal) |
| `app/qdrant_service.py` | Qdrant client factory and collection management |
| `app/chunker.py` | Markdown chunking by headers with metadata preservation |
| `app/ingest.py` | Data ingestion pipeline (read → chunk → embed → upsert) |
| `app/config.py` | Centralized configuration with env var support and validation |
| `app/schemas.py` | Pydantic request/response models with input validation |

### Middleware Stack

Execution order (first to last): RequestId → Logging → SecurityHeaders → RateLimit → CORS → handler.

### Database Schema

```sql
-- Conversation threads
threads (id TEXT PK, created_at TEXT)

-- Messages with user/language metadata
messages (id TEXT PK, thread_id TEXT FK, role TEXT, content TEXT,
          user_id TEXT, language TEXT, created_at TEXT)

-- External API response cache (per thread+user, unique constraint)
user_data_cache (id TEXT PK, thread_id TEXT FK, user_id TEXT,
                 data_json TEXT, created_at TEXT)
```

SQLite PRAGMAs: WAL mode, foreign keys, busy_timeout=5000, cache_size=8MB, synchronous=NORMAL.

---

## Prerequisites

1. **Python 3.10+**
2. **Ollama** — running and accessible. Required models:
   - `devstral:latest` (24B parameter LLM for generation and classification)
   - `nomic-embed-text` (embedding model, 768 dimensions)
3. **Qdrant** — vector database, default port 6333
4. **(Optional) NVIDIA GPU** — for Ollama inference acceleration

### Install Ollama Models

```bash
ollama pull devstral:latest
ollama pull nomic-embed-text
```

---

## Quick Start

### Option 1: Local Development

```bash
# 1. Clone the repository
git clone <repository-url>
cd devstral24b

# 2. Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Start Ollama and Qdrant (must be running)
# Ollama: typically on http://localhost:11434
# Qdrant: typically on http://localhost:6333

# 4. Ingest documentation into Qdrant
python -m app.ingest

# 5. Start the server
uvicorn app.main:app --reload
```

The server starts on `http://localhost:8000` by default.

### Option 2: Docker Compose

```bash
docker compose up -d
```

This starts three services:
- **app** — the chatbot API on port 2024
- **qdrant** — vector database on port 6333
- **ollama** — LLM server on port 11434 (with GPU support)

After services are healthy, ingest data:

```bash
docker compose exec app python -m app.ingest
```

---

## API Reference

### Create Thread

```
POST /api/chat/threads
```

**Response (201):**
```json
{
  "threadId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

### Send Message (Non-Streaming)

```
POST /api/chat/threads/{threadId}/messages
Content-Type: application/json
```

**Request Body:**
```json
{
  "message": "What schemes can I apply for?",
  "authToken": "",
  "userId": "",
  "language": "en"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `message` | string | Yes | User's question (1-10000 chars, not blank) |
| `authToken` | string | No | Bearer token for authenticated requests |
| `userId` | string | No | User ID (alphanumeric, `.`, `-`, `_`, max 100 chars) |
| `language` | string | No | ISO 639-1 code: `en`, `kn`, `hi`, `ta`, `te`, `ml`, `mr` |

**Response (201):**
```json
{
  "threadId": "a1b2c3d4-...",
  "messageId": "f8e7d6c5-...",
  "answer": "**Shrama Sahayak here!** Based on the available schemes..."
}
```

### Send Message (Streaming)

```
POST /api/chat/threads/{threadId}/messages/stream
Content-Type: application/json
```

Same request body as non-streaming. Returns `text/event-stream`:

```
data: {"event": "chunk", "content": "**Shrama"}
data: {"event": "chunk", "content": " Sahayak"}
data: {"event": "chunk", "content": " here!**"}
...
data: {"event": "done", "threadId": "...", "messageId": "...", "fullAnswer": "..."}
```

On error:
```
data: {"event": "error", "code": 504, "detail": "Response timed out"}
```

### Get Messages

```
GET /api/chat/threads/{threadId}/messages?limit=50&offset=0
```

| Param | Default | Range | Description |
|---|---|---|---|
| `limit` | 50 | 1-200 | Messages per page |
| `offset` | 0 | 0+ | Skip N messages |

**Response (200):**
```json
{
  "threadId": "...",
  "messages": [
    {"id": "...", "thread_id": "...", "role": "user", "content": "...", "created_at": "..."},
    {"id": "...", "thread_id": "...", "role": "assistant", "content": "...", "created_at": "..."}
  ],
  "total": 12,
  "limit": 50,
  "offset": 0
}
```

### Health Checks

| Endpoint | Purpose | Response |
|---|---|---|
| `GET /` | Liveness probe | `{"status": "ok"}` (always 200) |
| `GET /health` | Liveness probe (alias) | `{"status": "ok"}` (always 200) |
| `GET /ready` | Readiness probe | `{"status": "ready", "checks": {"database": "ok", "qdrant": "ok", "ollama": "ok"}}` (200 or 503) |

---

## Configuration

All configuration is in `app/config.py` via the `Settings` class. Every setting can be overridden with environment variables.

### Ollama

| Variable | Default | Description |
|---|---|---|
| `ENVIRONMENT` | `host` | `host` or `docker` — changes Ollama/Qdrant hostnames |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `devstral:latest` | LLM model for generation |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_TIMEOUT` | `120.0` | Non-streaming request timeout (seconds) |
| `OLLAMA_STREAM_TIMEOUT` | `300.0` | Streaming request timeout (seconds) |

### LLM Generation Parameters

| Variable | Default | Description |
|---|---|---|
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature |
| `LLM_TOP_P` | `0.9` | Nucleus sampling threshold |
| `LLM_TOP_K` | `40` | Top-k sampling |
| `LLM_REPEAT_PENALTY` | `1.1` | Repetition penalty |

### Qdrant

| Variable | Default | Description |
|---|---|---|
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `COLLECTION_NAME` | `ksk_docs` | Vector collection name |
| `VECTOR_SIZE` | `768` | Embedding dimensions |

### Retrieval

| Variable | Default | Description |
|---|---|---|
| `RETRIEVAL_TOP_K` | `5` | Number of chunks to retrieve |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.35` | Minimum cosine similarity score |

### Chunking

| Variable | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `1000` | Max characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `DATA_PATH` | `data/ksk.md` | Source markdown file |
| `INGEST_CONCURRENCY` | `5` | Parallel embedding tasks during ingest |

### Database

| Variable | Default | Description |
|---|---|---|
| `DATABASE_PATH` | `data/chat.db` | SQLite database file path |
| `MAX_HISTORY_MESSAGES` | `10` | Conversation history for unauthenticated users |
| `AUTHENTICATED_HISTORY_MESSAGES` | `6` | Conversation history for authenticated users |

### External API

| Variable | Default | Description |
|---|---|---|
| `BACKEND_API_URL` | `https://apikbocwwb.karnataka.gov.in/preprod/api` | Karnataka govt API base URL |
| `EXTERNAL_API_TIMEOUT` | `15.0` | Timeout for external API calls (seconds) |

### Rate Limiting

| Variable | Default | Description |
|---|---|---|
| `RATE_LIMIT_WINDOW` | `60` | Sliding window size (seconds) |
| `RATE_LIMIT_MAX_REQUESTS` | `30` | Max POST requests per window per IP |
| `MAX_TRACKED_IPS` | `50000` | Max IPs tracked before forced sweep |

### Security

| Variable | Default | Description |
|---|---|---|
| `TRUST_PROXY_HEADERS` | `false` | Trust X-Forwarded-For for real client IP |
| `CORS_ORIGINS` | `` | Comma-separated allowed origins |

### Response Limits

| Variable | Default | Description |
|---|---|---|
| `MAX_ANSWER_LENGTH` | `50000` | Max characters in LLM response |
| `MAX_RESPONSE_SIZE` | `100000` | Max bytes in raw Ollama response |

### Retention

| Variable | Default | Description |
|---|---|---|
| `MESSAGE_RETENTION_DAYS` | `90` | Delete messages older than N days |
| `CACHE_RETENTION_DAYS` | `7` | Delete cached user data older than N days |

### Other

| Variable | Default | Description |
|---|---|---|
| `MAX_THREAD_LOCKS` | `10000` | Max concurrent thread locks (LRU eviction) |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## Security

### Rate Limiting

- **IP-based sliding window** — tracks timestamps per IP, configurable window and max requests
- **Differentiated limits** — POST endpoints: 30/min, GET endpoints: 90/min
- **Memory exhaustion protection** — max tracked IPs cap with forced sweep
- **Reverse proxy support** — optional X-Forwarded-For trust via `TRUST_PROXY_HEADERS=true`

### Security Headers

Every response includes:
- `Strict-Transport-Security: max-age=31536000; includeSubDomains`
- `Content-Security-Policy: default-src 'none'`
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy: camera=(), microphone=(), geolocation=()`

### Input Validation

- **Message**: 1-10000 characters, must not be blank/whitespace-only (Pydantic)
- **userId**: regex `^[a-zA-Z0-9_.\-]{1,100}$` — prevents injection
- **threadId**: UUID v4 format validated on every endpoint
- **language**: validated against whitelist of ISO 639-1 codes
- **authToken**: max 500 characters

### Auth Token Protection

External API error handlers log only exception type/message, never full tracebacks that could contain authorization headers.

### Request Tracing

Every request receives a unique `X-Request-ID` header (generated or forwarded from client) for log correlation.

---

## Testing

The `tests/` directory contains automated API test suites:

| File | Description |
|---|---|
| `tests/helpers.py` | Shared utilities: HTTP helpers, `ResultWriter` for output, timing |
| `tests/test_single_thread.py` | 15 English + 15 Kannada test cases per thread (general, ecard, status, auth/unauth) |
| `tests/test_concurrent_users.py` | 3 concurrent users (A, B, C), 3 rounds via asyncio.gather |
| `tests/test_edge_cases.py` | 16 edge case categories (invalid UUID, XSS, SQL injection, rate limiting, streaming, etc.) |

### Running Tests

The server must be running before executing tests.

```bash
# Single thread tests
python tests/test_single_thread.py

# Concurrent user tests
python tests/test_concurrent_users.py

# Edge case tests
python tests/test_edge_cases.py
```

Override the server URL with `TEST_BASE_URL`:
```bash
TEST_BASE_URL=http://localhost:2024 python tests/test_single_thread.py
```

### Test Output

Results are saved to `test_results/` as `.txt` files with payload, response, and turnaround time for each API call. File names are printed to the terminal during execution.

---

## Docker Deployment

### docker-compose.yml

Three services with health checks and restart policies:

| Service | Image | Port | Notes |
|---|---|---|---|
| `app` | Built from `Dockerfile` | 2024 | Depends on qdrant + ollama healthy |
| `qdrant` | `qdrant/qdrant` | 6333 (localhost only) | Persistent volume for storage |
| `ollama` | `ollama/ollama` | 11434 (localhost only) | GPU passthrough, persistent model volume |

### Dockerfile

- Base: `python:3.11-slim`
- Non-root user (`appuser`) for security
- Copies `app/` and `data/` directories
- Runs uvicorn on port 2024

### GPU Support

The `ollama` service in docker-compose.yml is configured with NVIDIA GPU passthrough:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

Remove this section if running on CPU only.

---

## Project Structure

```
devstral24b/
├── app/
│   ├── __init__.py
│   ├── config.py            # Centralized configuration + validation
│   ├── main.py              # FastAPI app, middleware, endpoints
│   ├── rag.py               # Intent classification, prompts, RAG pipeline
│   ├── ollama_client.py     # Async Ollama HTTP client
│   ├── database.py          # SQLite schema, CRUD, cache, cleanup
│   ├── external_api.py      # Karnataka govt API integration
│   ├── qdrant_service.py    # Qdrant client factory
│   ├── chunker.py           # Markdown chunking with metadata
│   ├── ingest.py            # Data ingestion pipeline
│   └── schemas.py           # Pydantic request/response models
├── data/
│   └── ksk.md               # Source documentation (welfare schemes)
├── tests/
│   ├── helpers.py            # Shared test utilities
│   ├── test_single_thread.py # Single-thread API tests
│   ├── test_concurrent_users.py # Concurrent user tests
│   └── test_edge_cases.py    # Edge case tests
├── docker-compose.yml        # Multi-service Docker setup
├── Dockerfile                # Application container
├── requirements.txt          # Python dependencies
├── .gitignore
├── .dockerignore
└── CLAUDE.md                 # AI assistant context file
```
