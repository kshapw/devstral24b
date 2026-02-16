import logging
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
    """)
    # Migration: add columns if they don't exist (safe for existing databases)
    for col in ("user_id TEXT NOT NULL DEFAULT ''", "language TEXT NOT NULL DEFAULT ''"):
        try:
            await db.execute(f"ALTER TABLE messages ADD COLUMN {col}")
        except Exception:
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
            "WHERE thread_id = ? ORDER BY created_at ASC",
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


async def get_recent_thread_messages(
    db: aiosqlite.Connection, thread_id: str, limit: int
) -> list[dict]:
    """Fetch only the most recent `limit` messages for a thread, ordered chronologically."""
    logger.debug("Fetching last %d messages for thread %s", limit, thread_id)
    try:
        cursor = await db.execute(
            "SELECT id, thread_id, role, content, created_at FROM "
            "(SELECT id, thread_id, role, content, created_at FROM messages "
            "WHERE thread_id = ? ORDER BY created_at DESC LIMIT ?) "
            "ORDER BY created_at ASC",
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
