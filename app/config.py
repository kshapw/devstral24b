import logging
import os

logger = logging.getLogger(__name__)


class Settings:
    # Environment mode: "host" or "docker"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "host")

    # -------- Ollama --------
    OLLAMA_URL: str = os.getenv(
        "OLLAMA_URL",
        "http://ollama:11434" if ENVIRONMENT == "docker" else "http://localhost:11434",
    )
    LLM_MODEL: str = os.getenv("LLM_MODEL", "devstral:24b")
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")

    # -------- Qdrant --------
    QDRANT_HOST: str = os.getenv(
        "QDRANT_HOST",
        "qdrant" if ENVIRONMENT == "docker" else "localhost",
    )

    QDRANT_PORT: int = 6333
    COLLECTION_NAME: str = "ksk_docs"
    VECTOR_SIZE: int = 768

    # -------- SQLite --------
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "data/chat.db")

    # -------- History --------
    MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))

    # -------- Timeouts --------
    OLLAMA_TIMEOUT: float = float(os.getenv("OLLAMA_TIMEOUT", "120.0"))
    OLLAMA_STREAM_TIMEOUT: float = float(os.getenv("OLLAMA_STREAM_TIMEOUT", "300.0"))

    # -------- LLM Generation Parameters --------
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", "0.9"))
    LLM_TOP_K: int = int(os.getenv("LLM_TOP_K", "40"))
    LLM_REPEAT_PENALTY: float = float(os.getenv("LLM_REPEAT_PENALTY", "1.1"))

    # -------- Retrieval Parameters --------
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    RETRIEVAL_SCORE_THRESHOLD: float = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.35"))

    # -------- Chunking Parameters --------
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # -------- Ingest Concurrency --------
    INGEST_CONCURRENCY: int = int(os.getenv("INGEST_CONCURRENCY", "5"))

    # -------- Rate Limiting --------
    RATE_LIMIT_WINDOW: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
    RATE_LIMIT_MAX_REQUESTS: int = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "30"))

    # -------- Thread Lock Cache --------
    MAX_THREAD_LOCKS: int = int(os.getenv("MAX_THREAD_LOCKS", "10000"))

    # -------- Logging --------
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # -------- Data --------
    DATA_PATH: str = os.getenv("DATA_PATH", "data/ksk.md")


settings = Settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger.info("Environment: %s", settings.ENVIRONMENT)
logger.info("Ollama URL: %s", settings.OLLAMA_URL)
logger.info("Qdrant Host: %s", settings.QDRANT_HOST)
