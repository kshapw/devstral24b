import asyncio
import json
import logging
import os
import re
import time
import uuid as _uuid
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
    cleanup_old_data,
    ensure_thread,
    get_db,
    get_paginated_thread_messages,
    get_recent_thread_messages,
    init_db,
    thread_exists,
)
from app.ollama_client import OllamaClient
from app.qdrant_service import get_qdrant_client
from app.rag import answer, answer_stream, classify_and_prepare
from app.schemas import (
    HealthResponse,
    MessageRequest,
    MessageResponse,
    ThreadResponse,
)

logger = logging.getLogger(__name__)

# Thread ID pattern: accepts UUID v4 or thread-<digits> format
_UUID_RE = re.compile(
    r"^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|thread-\d+)$", re.IGNORECASE
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
        # Relaxed CSP for Swagger UI to work
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "img-src 'self' data: https://fastapi.tiangolo.com;"
        )
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
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

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, respecting X-Forwarded-For if proxy headers are trusted."""
        if settings.TRUST_PROXY_HEADERS:
            forwarded = request.headers.get("X-Forwarded-For", "")
            if forwarded:
                return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _do_sweep(self, now: float) -> None:
        """Remove all stale IPs from the hits dict."""
        stale_ips = [
            ip for ip, ts in self._hits.items()
            if not ts or all(now - t >= self.window for t in ts)
        ]
        for ip in stale_ips:
            del self._hits[ip]
        if stale_ips:
            logger.debug("Swept %d stale IPs from rate limiter", len(stale_ips))

    async def dispatch(self, request: Request, call_next):
        # Apply different limits: stricter for POST, lighter for GET
        if request.method == "GET":
            max_requests = self.max_requests * 3  # 90 per window for GET
        elif request.method == "POST":
            max_requests = self.max_requests  # 30 per window for POST
        else:
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        now = time()

        # Clean old entries for this IP
        timestamps = self._hits.get(client_ip, [])
        timestamps = [t for t in timestamps if now - t < self.window]

        if len(timestamps) >= max_requests:
            logger.warning("Rate limit exceeded for IP %s (%s)", client_ip, request.method)
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please try again later."},
            )

        timestamps.append(now)
        self._hits[client_ip] = timestamps

        # Cap tracked IPs to prevent memory exhaustion
        if len(self._hits) > settings.MAX_TRACKED_IPS:
            self._do_sweep(now)

        # Periodically sweep all stale IPs (every 50 requests)
        self._sweep_counter += 1
        if self._sweep_counter >= 50:
            self._sweep_counter = 0
            self._do_sweep(now)

        return await call_next(request)


# ---------------------------------------------------------------------------
# Request ID middleware (log correlation)
# ---------------------------------------------------------------------------
class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(_uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ---------------------------------------------------------------------------
# Request logging middleware (timing)
# ---------------------------------------------------------------------------
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time()
        response = await call_next(request)
        elapsed = (time() - start) * 1000
        logger.info(
            "%s %s %d %.0fms",
            request.method, request.url.path, response.status_code, elapsed,
        )
        return response


# ---------------------------------------------------------------------------
# Background periodic cleanup
# ---------------------------------------------------------------------------
async def _periodic_cleanup(app_instance: FastAPI) -> None:
    """Run retention cleanup hourly in the background."""
    while True:
        await asyncio.sleep(3600)  # hourly
        try:
            msgs, cache = await cleanup_old_data(app_instance.state.db)
            if msgs or cache:
                logger.info("Periodic cleanup: %d messages, %d cache entries removed", msgs, cache)
        except Exception:
            logger.error("Periodic cleanup failed", exc_info=True)


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
        # Separate client for external Karnataka API (needs verify=False)
        app.state.ext_http_client = httpx.AsyncClient(timeout=30, verify=False)
        logger.info("Ollama client and external API client created")
    except Exception:
        logger.critical("Failed to create HTTP clients", exc_info=True)
        raise

    # Warmup the model (loads it into GPU memory)
    try:
        logger.info("Warming up Ollama model...")
        # Simple generation to trigger model load
        await app.state.ollama.generate("Hello")
        logger.info("Ollama model warmup successful")
    except Exception:
        logger.warning("Ollama model warmup failed (non-fatal)", exc_info=True)

    try:
        app.state.db = await get_db()
        await init_db(app.state.db)
        logger.info("Database initialized")
    except Exception:
        logger.critical("Failed to initialize database", exc_info=True)
        raise

    # Run initial retention cleanup
    try:
        msgs, cache = await cleanup_old_data(app.state.db)
        if msgs or cache:
            logger.info("Startup cleanup: %d messages, %d cache entries removed", msgs, cache)
    except Exception:
        logger.error("Startup cleanup failed (non-fatal)", exc_info=True)

    app.state.thread_locks: OrderedDict[str, asyncio.Lock] = OrderedDict()

    # Launch background cleanup task
    cleanup_task = asyncio.create_task(_periodic_cleanup(app))
    logger.info("Application startup complete")

    yield

    logger.info("Shutting down application")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    await app.state.db.close()
    await app.state.qdrant.close()
    await app.state.http_client.aclose()
    await app.state.ext_http_client.aclose()
    logger.info("Application shutdown complete")


app = FastAPI(lifespan=lifespan)

# --- Middleware stack (order matters: last added = first executed) ---
# Execution order: RequestId → Logging → Security → RateLimit → CORS → handler
app.add_middleware(RequestIdMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware)

_raw_origins = os.getenv("CORS_ORIGINS", "")
_cors_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
if not _cors_origins:
    _cors_origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],  # Allow all methods including OPTIONS
    allow_headers=["*"],  # Allow all headers (Authorization, X-Request-ID, etc.)
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
    """Liveness probe — always returns 200 if process is running."""
    return HealthResponse(status="ok")


@app.get("/health", response_model=HealthResponse)
async def health_check_alias():
    """Liveness probe alias at /health."""
    return HealthResponse(status="ok")


@app.get("/ready")
async def readiness_check():
    """Readiness probe — checks all dependencies."""
    checks: dict[str, str] = {}
    # Database
    try:
        cursor = await app.state.db.execute("SELECT 1")
        await cursor.fetchone()
        checks["database"] = "ok"
    except Exception:
        checks["database"] = "failed"
    # Qdrant
    try:
        await app.state.qdrant.get_collections()
        checks["qdrant"] = "ok"
    except Exception:
        checks["qdrant"] = "failed"
    # Ollama
    try:
        resp = await app.state.http_client.get(
            f"{settings.OLLAMA_URL}/api/tags", timeout=5.0,
        )
        checks["ollama"] = "ok" if resp.status_code == 200 else "failed"
    except Exception:
        checks["ollama"] = "failed"

    all_ok = all(v == "ok" for v in checks.values())
    status_code = 200 if all_ok else 503
    return JSONResponse(
        content={
            "status": "ready" if all_ok else "degraded",
            "checks": checks,
        },
        status_code=status_code,
    )


@app.post("/api/chat/threads", status_code=200)
async def create_chat_thread():
    """Lightweight thread creation — returns a timestamp-based ID, no DB write."""
    import time
    thread_id = f"thread-{int(time.time())}"
    logger.info("Created chat thread %s", thread_id)
    return {"id": thread_id}


@app.get("/api/chat/threads/{threadId}/messages", status_code=200)
async def get_messages(
    threadId: str,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: aiosqlite.Connection = Depends(get_db_dep),
):
    """Retrieve conversation history for a thread with pagination."""
    _validate_thread_id(threadId)

    messages, total = await get_paginated_thread_messages(db, threadId, limit, offset)
    return {
        "threadId": threadId,
        "messages": messages,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.post(
    "/api/chat/threads/{threadId}/messages",
    response_model=MessageResponse,
    status_code=200,
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
        await ensure_thread(db, threadId)

        await add_message(
            db, threadId, "user", body.message,
            user_id=body.userId, language=body.language,
        )
        logger.info("User message added to thread %s", threadId)

        # Determine history limit based on authentication status
        is_authenticated = bool(body.userId and body.authToken)
        history_limit = (
            settings.AUTHENTICATED_HISTORY_MESSAGES if is_authenticated
            else settings.MAX_HISTORY_MESSAGES
        )

        # Fetch recent messages for history (+ 1 to account for the message just added)
        messages = await get_recent_thread_messages(
            db, threadId, limit=history_limit + 1
        )
        # Exclude the last message (just added) to form history
        history = messages[:-1] if len(messages) > 1 else None

        # Classify intent + pre-fetch data under lock (prevents race conditions)
        intent, prefetched_user_data = await classify_and_prepare(
            ollama=ollama,
            message=body.message,
            user_id=body.userId,
            auth_token=body.authToken,
            db=db,
            http_client=app.state.ext_http_client,
            thread_id=threadId,
        )

        try:
            result = await answer(
                question=body.message,
                qdrant=qdrant,
                ollama=ollama,
                history=history,
                language=body.language,
                intent=intent,
                prefetched_user_data=prefetched_user_data,
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
            await add_message(
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
            "Assistant response added to thread %s", threadId
        )
        return MessageResponse(
            message=body.message,
            reply=result,
            options=[],
            audioUrl=None,
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
    # Validate thread, save user message, classify intent, and pre-fetch data
    # ALL under the thread lock to prevent race conditions on cache read/write.
    async with get_thread_lock(threadId):
        await ensure_thread(db, threadId)

        await add_message(
            db, threadId, "user", body.message,
            user_id=body.userId, language=body.language,
        )
        logger.info("User message added to thread %s (stream)", threadId)

        # Determine history limit based on authentication status
        is_authenticated = bool(body.userId and body.authToken)
        history_limit = (
            settings.AUTHENTICATED_HISTORY_MESSAGES if is_authenticated
            else settings.MAX_HISTORY_MESSAGES
        )

        messages = await get_recent_thread_messages(
            db, threadId, limit=history_limit + 1
        )
        history = messages[:-1] if len(messages) > 1 else None

        # Classify intent + pre-fetch data under lock (prevents race conditions)
        intent, prefetched_user_data = await classify_and_prepare(
            ollama=ollama,
            message=body.message,
            user_id=body.userId,
            auth_token=body.authToken,
            db=db,
            http_client=app.state.ext_http_client,
            thread_id=threadId,
        )

    # Lock released — streaming happens outside the lock using pre-computed data
    async def event_generator():
        full_answer_parts: list[str] = []
        try:
            async with asyncio.timeout(settings.OLLAMA_STREAM_TIMEOUT):
                async for chunk in answer_stream(
                    question=body.message,
                    qdrant=qdrant,
                    ollama=ollama,
                    history=history,
                    language=body.language,
                    intent=intent,
                    prefetched_user_data=prefetched_user_data,
                ):
                    full_answer_parts.append(chunk)
                    event_data = json.dumps({"event": "chunk", "content": chunk})
                    yield f"data: {event_data}\n\n"
        except TimeoutError:
            logger.error("Streaming timed out for thread %s", threadId)
            error_data = json.dumps({"event": "error", "code": 504, "detail": "Response timed out"})
            yield f"data: {error_data}\n\n"
            return
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
