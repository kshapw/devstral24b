import asyncio
import json
import logging
import os
import re
from collections import OrderedDict
from contextlib import asynccontextmanager
from time import time

import aiosqlite
import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from qdrant_client import AsyncQdrantClient
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings
from app.database import (
    add_message,
    create_thread,
    get_db,
    get_recent_thread_messages,
    get_thread_messages,
    init_db,
    thread_exists,
)
from app.ollama_client import OllamaClient
from app.qdrant_service import get_qdrant_client
from app.rag import answer, answer_stream
from app.schemas import (
    HealthResponse,
    MessageRequest,
    MessageResponse,
    ThreadResponse,
)

logger = logging.getLogger(__name__)

# UUID v4 pattern for threadId validation
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)



def _validate_thread_id(thread_id: str) -> None:
    """Raise 400 if thread_id is not a valid UUID."""
    if not _UUID_RE.match(thread_id):
        raise HTTPException(status_code=400, detail="Invalid thread ID format")


# ---------------------------------------------------------------------------
# Security headers middleware
# ---------------------------------------------------------------------------
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = "default-src 'none'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


# ---------------------------------------------------------------------------
# Rate limiting middleware (in-memory, IP-based sliding window)
# ---------------------------------------------------------------------------
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int | None = None, window: int | None = None):
        super().__init__(app)
        self.max_requests = max_requests if max_requests is not None else settings.RATE_LIMIT_MAX_REQUESTS
        self.window = window if window is not None else settings.RATE_LIMIT_WINDOW
        self._hits: dict[str, list[float]] = {}
        self._sweep_counter: int = 0

    async def dispatch(self, request: Request, call_next):
        # Only rate-limit mutating endpoints
        if request.method != "POST":
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time()

        # Clean old entries for this IP
        timestamps = self._hits.get(client_ip, [])
        timestamps = [t for t in timestamps if now - t < self.window]

        if len(timestamps) >= self.max_requests:
            logger.warning("Rate limit exceeded for IP %s", client_ip)
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please try again later."},
            )

        timestamps.append(now)
        self._hits[client_ip] = timestamps

        # Periodically sweep all stale IPs (every 100 requests)
        self._sweep_counter += 1
        if self._sweep_counter >= 100:
            self._sweep_counter = 0
            stale_ips = [
                ip for ip, ts in self._hits.items()
                if not ts or all(now - t >= self.window for t in ts)
            ]
            for ip in stale_ips:
                del self._hits[ip]
            if stale_ips:
                logger.debug("Swept %d stale IPs from rate limiter", len(stale_ips))

        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application")

    try:
        app.state.qdrant = get_qdrant_client()
        logger.info("Qdrant client created")
    except Exception:
        logger.critical("Failed to create Qdrant client", exc_info=True)
        raise

    try:
        app.state.http_client = httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT)
        app.state.ollama = OllamaClient(client=app.state.http_client)
        logger.info("Ollama client created")
    except Exception:
        logger.critical("Failed to create Ollama HTTP client", exc_info=True)
        raise

    try:
        app.state.db = await get_db()
        await init_db(app.state.db)
        logger.info("Database initialized")
    except Exception:
        logger.critical("Failed to initialize database", exc_info=True)
        raise

    app.state.thread_locks: OrderedDict[str, asyncio.Lock] = OrderedDict()
    logger.info("Application startup complete")

    yield

    logger.info("Shutting down application")
    await app.state.db.close()
    await app.state.qdrant.close()
    await app.state.http_client.aclose()
    logger.info("Application shutdown complete")


app = FastAPI(lifespan=lifespan)

# --- Middleware stack (order matters: last added = first executed) ---
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else [],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    logger.error(
        "Unhandled exception on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# Dependencies
def get_qdrant_dep() -> AsyncQdrantClient:
    return app.state.qdrant


def get_ollama_dep() -> OllamaClient:
    return app.state.ollama


def get_db_dep() -> aiosqlite.Connection:
    return app.state.db


def get_thread_lock(thread_id: str) -> asyncio.Lock:
    """Return (or create) an asyncio.Lock for the given thread.

    Uses an OrderedDict as a bounded LRU cache to prevent memory exhaustion.
    """
    locks: OrderedDict[str, asyncio.Lock] = app.state.thread_locks
    if thread_id in locks:
        locks.move_to_end(thread_id)
        return locks[thread_id]
    # Evict oldest if at capacity
    while len(locks) >= settings.MAX_THREAD_LOCKS:
        evicted_id, _ = locks.popitem(last=False)
        logger.debug("Evicted thread lock for %s (LRU)", evicted_id)
    lock = asyncio.Lock()
    locks[thread_id] = lock
    return lock


@app.get("/", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok")


@app.post("/api/chat/threads", response_model=ThreadResponse, status_code=201)
async def create_chat_thread(
    db: aiosqlite.Connection = Depends(get_db_dep),
):
    thread_id = await create_thread(db)
    logger.info("Created chat thread %s", thread_id)
    return ThreadResponse(threadId=thread_id)


@app.get("/api/chat/threads/{threadId}/messages", status_code=200)
async def get_messages(
    threadId: str,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: aiosqlite.Connection = Depends(get_db_dep),
):
    """Retrieve conversation history for a thread with pagination."""
    _validate_thread_id(threadId)
    if not await thread_exists(db, threadId):
        raise HTTPException(status_code=404, detail="Thread not found")

    messages = await get_thread_messages(db, threadId)
    paginated = messages[offset : offset + limit]
    return {
        "threadId": threadId,
        "messages": paginated,
        "total": len(messages),
        "limit": limit,
        "offset": offset,
    }


@app.post(
    "/api/chat/threads/{threadId}/messages",
    response_model=MessageResponse,
    status_code=201,
)
async def send_message(
    threadId: str,
    body: MessageRequest,
    db: aiosqlite.Connection = Depends(get_db_dep),
    qdrant: AsyncQdrantClient = Depends(get_qdrant_dep),
    ollama: OllamaClient = Depends(get_ollama_dep),
):
    _validate_thread_id(threadId)
    async with get_thread_lock(threadId):
        if not await thread_exists(db, threadId):
            raise HTTPException(status_code=404, detail="Thread not found")

        await add_message(
            db, threadId, "user", body.message,
            user_id=body.userId, language=body.language,
        )
        logger.info("User message added to thread %s", threadId)

        # Fetch recent messages for history (+ 1 to account for the message just added)
        messages = await get_recent_thread_messages(
            db, threadId, limit=settings.MAX_HISTORY_MESSAGES + 1
        )
        # Exclude the last message (just added) to form history
        history = messages[:-1] if len(messages) > 1 else None

        try:
            result = await answer(
                question=body.message,
                qdrant=qdrant,
                ollama=ollama,
                history=history,
                language=body.language,
            )
        except Exception:
            logger.error(
                "RAG pipeline failed for thread %s", threadId, exc_info=True
            )
            raise HTTPException(
                status_code=502,
                detail="Failed to generate a response. Please try again later.",
            )

        try:
            assistant_msg_id = await add_message(
                db, threadId, "assistant", result,
                user_id=body.userId, language=body.language,
            )
        except Exception:
            logger.error(
                "Failed to save assistant response to thread %s", threadId, exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail="Response generated but failed to save. Please try again.",
            )

        logger.info(
            "Assistant response %s added to thread %s", assistant_msg_id, threadId
        )
        return MessageResponse(
            threadId=threadId,
            messageId=assistant_msg_id,
            answer=result,
        )


@app.post("/api/chat/threads/{threadId}/messages/stream", status_code=200)
async def send_message_stream(
    threadId: str,
    body: MessageRequest,
    db: aiosqlite.Connection = Depends(get_db_dep),
    qdrant: AsyncQdrantClient = Depends(get_qdrant_dep),
    ollama: OllamaClient = Depends(get_ollama_dep),
):
    """Streaming variant — returns Server-Sent Events."""
    _validate_thread_id(threadId)
    # Validate thread and save user message under lock, then release before streaming
    async with get_thread_lock(threadId):
        if not await thread_exists(db, threadId):
            raise HTTPException(status_code=404, detail="Thread not found")

        await add_message(
            db, threadId, "user", body.message,
            user_id=body.userId, language=body.language,
        )
        logger.info("User message added to thread %s (stream)", threadId)

        messages = await get_recent_thread_messages(
            db, threadId, limit=settings.MAX_HISTORY_MESSAGES + 1
        )
        history = messages[:-1] if len(messages) > 1 else None

    async def event_generator():
        full_answer_parts: list[str] = []
        try:
            async for chunk in answer_stream(
                question=body.message,
                qdrant=qdrant,
                ollama=ollama,
                history=history,
                language=body.language,
            ):
                full_answer_parts.append(chunk)
                event_data = json.dumps({"event": "chunk", "content": chunk})
                yield f"data: {event_data}\n\n"
        except Exception:
            logger.error(
                "Streaming RAG pipeline failed for thread %s",
                threadId,
                exc_info=True,
            )
            error_data = json.dumps({"event": "error", "code": 502, "detail": "Generation failed"})
            yield f"data: {error_data}\n\n"
            return

        # Stream complete — save the full answer to the database
        full_answer = "".join(full_answer_parts)
        assistant_msg_id = None
        try:
            async with get_thread_lock(threadId):
                assistant_msg_id = await add_message(
                    db, threadId, "assistant", full_answer,
                    user_id=body.userId, language=body.language,
                )
        except Exception:
            logger.error(
                "Failed to save streamed response to thread %s (answer len=%d)",
                threadId, len(full_answer), exc_info=True,
            )

        done_data = json.dumps({
            "event": "done",
            "threadId": threadId,
            "messageId": assistant_msg_id,
            "fullAnswer": full_answer,
        })
        yield f"data: {done_data}\n\n"

        logger.info(
            "Streamed response %s to thread %s (%d chars)",
            assistant_msg_id, threadId, len(full_answer),
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
