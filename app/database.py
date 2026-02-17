import json
import logging
import sqlite3
import uuid

import aiosqlite

from app.config import settings

logger = logging.getLogger(__name__)


async def get_db() -> aiosqlite.Connection:
    logger.info("Connecting to SQLite database at %s", settings.DATABASE_PATH)
    try:
        db = await aiosqlite.connect(settings.DATABASE_PATH)
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")
        await db.execute("PRAGMA busy_timeout=5000")
        await db.execute("PRAGMA cache_size=-8000")
        await db.execute("PRAGMA synchronous=NORMAL")
        return db
    except Exception:
        logger.critical(
            "Failed to connect to SQLite database at %s", settings.DATABASE_PATH, exc_info=True,
        )
        raise


async def init_db(db: aiosqlite.Connection) -> None:
    logger.info("Initializing database schema")
    await db.executescript("""
        CREATE TABLE IF NOT EXISTS threads (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            user_id TEXT NOT NULL DEFAULT '',
            language TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (thread_id) REFERENCES threads(id)
        );
        CREATE INDEX IF NOT EXISTS idx_messages_thread_id ON messages(thread_id);

        CREATE TABLE IF NOT EXISTS user_data_cache (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            data_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (thread_id) REFERENCES threads(id)
        );
        CREATE INDEX IF NOT EXISTS idx_user_data_cache_thread
            ON user_data_cache(thread_id, user_id);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_user_data_cache_unique
            ON user_data_cache(thread_id, user_id);
    """)
    # Migration: add columns if they don't exist (safe for existing databases)
    for col in ("user_id TEXT NOT NULL DEFAULT ''", "language TEXT NOT NULL DEFAULT ''"):
        try:
            await db.execute(f"ALTER TABLE messages ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass  # Column already exists
    await db.commit()
    logger.info("Database schema initialized")


async def create_thread(db: aiosqlite.Connection) -> str:
    thread_id = str(uuid.uuid4())
    try:
        await db.execute("INSERT INTO threads (id) VALUES (?)", (thread_id,))
        await db.commit()
    except Exception:
        logger.error("Failed to create thread %s", thread_id, exc_info=True)
        raise
    logger.info("Created thread %s", thread_id)
    return thread_id


async def thread_exists(db: aiosqlite.Connection, thread_id: str) -> bool:
    try:
        cursor = await db.execute(
            "SELECT 1 FROM threads WHERE id = ?", (thread_id,)
        )
        row = await cursor.fetchone()
        return row is not None
    except Exception:
        logger.error("Failed to check thread existence for %s", thread_id, exc_info=True)
        raise


async def add_message(
    db: aiosqlite.Connection,
    thread_id: str,
    role: str,
    content: str,
    user_id: str = "",
    language: str = "",
) -> str:
    message_id = str(uuid.uuid4())
    try:
        await db.execute(
            "INSERT INTO messages (id, thread_id, role, content, user_id, language) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (message_id, thread_id, role, content, user_id, language),
        )
        await db.commit()
    except Exception:
        logger.error(
            "Failed to add %s message to thread %s", role, thread_id, exc_info=True,
        )
        raise
    logger.debug("Added %s message %s to thread %s", role, message_id, thread_id)
    return message_id


async def get_thread_messages(
    db: aiosqlite.Connection, thread_id: str
) -> list[dict]:
    try:
        cursor = await db.execute(
            "SELECT id, thread_id, role, content, created_at FROM messages "
            "WHERE thread_id = ? ORDER BY created_at ASC, rowid ASC",
            (thread_id,),
        )
        rows = await cursor.fetchall()
    except Exception:
        logger.error("Failed to fetch messages for thread %s", thread_id, exc_info=True)
        raise
    return [
        {
            "id": row[0],
            "thread_id": row[1],
            "role": row[2],
            "content": row[3],
            "created_at": row[4],
        }
        for row in rows
    ]


async def get_paginated_thread_messages(
    db: aiosqlite.Connection, thread_id: str, limit: int, offset: int
) -> tuple[list[dict], int]:
    """Fetch messages for a thread with SQL-level pagination. Returns (messages, total)."""
    logger.debug("Fetching messages for thread %s (limit=%d, offset=%d)", thread_id, limit, offset)
    try:
        count_cursor = await db.execute(
            "SELECT COUNT(*) FROM messages WHERE thread_id = ?",
            (thread_id,),
        )
        total = (await count_cursor.fetchone())[0]

        cursor = await db.execute(
            "SELECT id, thread_id, role, content, created_at FROM messages "
            "WHERE thread_id = ? ORDER BY created_at ASC, rowid ASC "
            "LIMIT ? OFFSET ?",
            (thread_id, limit, offset),
        )
        rows = await cursor.fetchall()
    except Exception:
        logger.error(
            "Failed to fetch paginated messages for thread %s", thread_id, exc_info=True,
        )
        raise
    messages = [
        {
            "id": row[0],
            "thread_id": row[1],
            "role": row[2],
            "content": row[3],
            "created_at": row[4],
        }
        for row in rows
    ]
    return messages, total


async def get_recent_thread_messages(
    db: aiosqlite.Connection, thread_id: str, limit: int
) -> list[dict]:
    """Fetch only the most recent `limit` messages for a thread, ordered chronologically."""
    logger.debug("Fetching last %d messages for thread %s", limit, thread_id)
    try:
        cursor = await db.execute(
            "SELECT id, thread_id, role, content, created_at FROM "
            "(SELECT id, thread_id, role, content, created_at, rowid FROM messages "
            "WHERE thread_id = ? ORDER BY created_at DESC, rowid DESC LIMIT ?) "
            "ORDER BY created_at ASC, rowid ASC",
            (thread_id, limit),
        )
        rows = await cursor.fetchall()
    except Exception:
        logger.error(
            "Failed to fetch recent messages for thread %s", thread_id, exc_info=True,
        )
        raise
    return [
        {
            "id": row[0],
            "thread_id": row[1],
            "role": row[2],
            "content": row[3],
            "created_at": row[4],
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# User data cache (for external API results)
# ---------------------------------------------------------------------------
async def get_cached_user_data(
    db: aiosqlite.Connection, thread_id: str, user_id: str
) -> dict | None:
    """Return cached user data for this thread+user, or None if not cached."""
    try:
        cursor = await db.execute(
            "SELECT data_json FROM user_data_cache "
            "WHERE thread_id = ? AND user_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (thread_id, user_id),
        )
        row = await cursor.fetchone()
    except Exception:
        logger.error(
            "Failed to fetch cached user data for thread %s, user %s",
            thread_id, user_id, exc_info=True,
        )
        return None
    if row is None:
        return None
    try:
        return json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        logger.error("Corrupt cached user data for thread %s, user %s", thread_id, user_id)
        return None


async def save_user_data(
    db: aiosqlite.Connection, thread_id: str, user_id: str, data: dict
) -> str:
    """Save fetched user data to cache. Returns the cache entry ID."""
    cache_id = str(uuid.uuid4())
    data_json = json.dumps(data, ensure_ascii=False, default=str)
    try:
        await db.execute(
            "INSERT OR REPLACE INTO user_data_cache (id, thread_id, user_id, data_json) "
            "VALUES (?, ?, ?, ?)",
            (cache_id, thread_id, user_id, data_json),
        )
        await db.commit()
    except Exception:
        logger.error(
            "Failed to save user data cache for thread %s, user %s",
            thread_id, user_id, exc_info=True,
        )
        raise
    logger.info("Cached user data %s for thread %s, user %s", cache_id, thread_id, user_id)
    return cache_id


# ---------------------------------------------------------------------------
# Retention cleanup
# ---------------------------------------------------------------------------
async def cleanup_old_data(
    db: aiosqlite.Connection,
    message_days: int | None = None,
    cache_days: int | None = None,
) -> tuple[int, int]:
    """Delete messages older than message_days and cache entries older than cache_days.

    Returns (messages_deleted, cache_entries_deleted).
    """
    message_days = message_days if message_days is not None else settings.MESSAGE_RETENTION_DAYS
    cache_days = cache_days if cache_days is not None else settings.CACHE_RETENTION_DAYS

    msg_result = await db.execute(
        "DELETE FROM messages WHERE created_at < datetime('now', ?)",
        (f"-{message_days} days",),
    )
    cache_result = await db.execute(
        "DELETE FROM user_data_cache WHERE created_at < datetime('now', ?)",
        (f"-{cache_days} days",),
    )
    await db.commit()

    msgs_deleted = msg_result.rowcount
    cache_deleted = cache_result.rowcount
    if msgs_deleted or cache_deleted:
        logger.info(
            "Retention cleanup: %d messages, %d cache entries removed",
            msgs_deleted, cache_deleted,
        )
    return msgs_deleted, cache_deleted
