# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG (Retrieval-Augmented Generation) chatbot for Karmika Seva Kendra (KSK) — a welfare assistance service for construction workers in Karnataka, India. The system ingests Markdown documentation about welfare schemes, stores vector embeddings in Qdrant, and answers questions using the Devstral 24B LLM via Ollama.

## Commands

### Setup
```bash
pip install -r requirements.txt
```

### Ingest data into Qdrant
```bash
python -m app.ingest
```
Reads `data/ksk.md`, chunks it by Markdown headers, embeds each chunk with `nomic-embed-text` via Ollama, and upserts into Qdrant collection `ksk_docs`.

### Run the server
```bash
uvicorn app.main:app --reload
```
Starts FastAPI on default port. Single endpoint: `POST /chat` with JSON body `{"question": "..."}`.

### Test scripts (no formal test framework)
```bash
python test_rag.py          # Single question through full RAG pipeline
python test_retrieval.py    # Test Qdrant retrieval only (no LLM generation)
python benchmark_rag.py     # Run 10 queries, measure response times, output to rag_performance.txt
```

## Architecture

**Request flow:** `POST /chat` → `app/main.py` → `app/rag.py:answer()` → `retrieve()` (embed query → Qdrant similarity search) → build prompt with context → `OllamaClient.generate()` → return response.

**Ingestion flow:** `app/ingest.py` → `app/chunker.py` (split by `#`/`##` headers via langchain) → `OllamaClient.embed()` → Qdrant upsert.

### Key modules
- **`app/main.py`** — FastAPI app with async lifespan managing Qdrant and httpx clients. Dependency injection via `app.state`.
- **`app/rag.py`** — `retrieve()` embeds the query and searches Qdrant; `answer()` builds a prompt with retrieved context and calls the LLM.
- **`app/ollama_client.py`** — Async wrapper around Ollama REST API (`/api/generate`, `/api/embeddings`). A `default_ollama` singleton exists for scripts; the server uses injected instances.
- **`app/qdrant_service.py`** — Factory for `AsyncQdrantClient` and collection creation (768-dim cosine vectors).
- **`app/chunker.py`** — Splits Markdown by `#` (section) and `##` (scheme) headers, prepends metadata to each chunk.
- **`app/config.py`** — `Settings` class with environment-aware config. Set `ENVIRONMENT=docker` for Docker networking, defaults to `host` (Ollama on port 11439, Qdrant on localhost:6333).

## External Services Required

Both must be running before starting the server or ingestion:
- **Ollama** — LLM server. Host mode expects port 11439 (mapped to internal 11434). Models needed: `devstral:24b` (generation), `nomic-embed-text` (embeddings).
- **Qdrant** — Vector database on port 6333.

## Configuration

All config in `app/config.py`. Key environment variables (all optional, have defaults):
- `ENVIRONMENT` — `"host"` (default) or `"docker"`. Changes Ollama/Qdrant hostnames.
- `LLM_MODEL` — defaults to `devstral:24b`
- `EMBED_MODEL` — defaults to `nomic-embed-text`

Dockerfile and docker-compose.yml exist but are currently empty placeholders.
