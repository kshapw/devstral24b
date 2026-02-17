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
    LLM_MODEL: str = os.getenv("LLM_MODEL", "devstral:latest")
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
    AUTHENTICATED_HISTORY_MESSAGES: int = int(os.getenv("AUTHENTICATED_HISTORY_MESSAGES", "6"))

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

    # -------- External Backend API --------
    BACKEND_API_URL: str = os.getenv(
        "BACKEND_API_URL", "https://apikbocwwb.karnataka.gov.in/preprod/api"
    )
    EXTERNAL_API_TIMEOUT: float = float(os.getenv("EXTERNAL_API_TIMEOUT", "15.0"))

    # -------- Rate Limiting --------
    RATE_LIMIT_WINDOW: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
    RATE_LIMIT_MAX_REQUESTS: int = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "30"))

    # -------- Thread Lock Cache --------
    MAX_THREAD_LOCKS: int = int(os.getenv("MAX_THREAD_LOCKS", "10000"))

    # -------- Logging --------
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # -------- Data --------
    DATA_PATH: str = os.getenv("DATA_PATH", "data/ksk.md")

    # -------- Cleanup / Retention --------
    MESSAGE_RETENTION_DAYS: int = int(os.getenv("MESSAGE_RETENTION_DAYS", "90"))
    CACHE_RETENTION_DAYS: int = int(os.getenv("CACHE_RETENTION_DAYS", "7"))

    # -------- Security --------
    TRUST_PROXY_HEADERS: bool = os.getenv("TRUST_PROXY_HEADERS", "false").lower() == "true"
    MAX_TRACKED_IPS: int = int(os.getenv("MAX_TRACKED_IPS", "50000"))

    # -------- Response Limits --------
    MAX_ANSWER_LENGTH: int = int(os.getenv("MAX_ANSWER_LENGTH", "50000"))
    MAX_RESPONSE_SIZE: int = int(os.getenv("MAX_RESPONSE_SIZE", "100000"))


settings = Settings()


def _validate_settings(s: Settings) -> None:
    """Validate all settings at startup. Raises AssertionError on invalid config."""
    assert s.OLLAMA_TIMEOUT > 0, "OLLAMA_TIMEOUT must be positive"
    assert s.OLLAMA_STREAM_TIMEOUT > 0, "OLLAMA_STREAM_TIMEOUT must be positive"
    assert s.RATE_LIMIT_MAX_REQUESTS > 0, "RATE_LIMIT_MAX_REQUESTS must be positive"
    assert s.RATE_LIMIT_WINDOW > 0, "RATE_LIMIT_WINDOW must be positive"
    assert 0 < s.RETRIEVAL_SCORE_THRESHOLD <= 1.0, "RETRIEVAL_SCORE_THRESHOLD must be in (0, 1]"
    assert s.MAX_HISTORY_MESSAGES > 0, "MAX_HISTORY_MESSAGES must be positive"
    assert s.CHUNK_SIZE > 0, "CHUNK_SIZE must be positive"
    assert s.CHUNK_OVERLAP >= 0, "CHUNK_OVERLAP must be non-negative"
    assert s.CHUNK_OVERLAP < s.CHUNK_SIZE, "CHUNK_OVERLAP must be less than CHUNK_SIZE"
    assert s.VECTOR_SIZE > 0, "VECTOR_SIZE must be positive"
    assert s.MAX_THREAD_LOCKS > 0, "MAX_THREAD_LOCKS must be positive"
    assert s.INGEST_CONCURRENCY > 0, "INGEST_CONCURRENCY must be positive"
    assert s.MESSAGE_RETENTION_DAYS > 0, "MESSAGE_RETENTION_DAYS must be positive"
    assert s.CACHE_RETENTION_DAYS > 0, "CACHE_RETENTION_DAYS must be positive"
    assert s.MAX_TRACKED_IPS > 0, "MAX_TRACKED_IPS must be positive"
    assert s.MAX_ANSWER_LENGTH > 0, "MAX_ANSWER_LENGTH must be positive"
    assert s.MAX_RESPONSE_SIZE > 0, "MAX_RESPONSE_SIZE must be positive"


_validate_settings(settings)

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger.info("Environment: %s", settings.ENVIRONMENT)
logger.info("Ollama URL: %s", settings.OLLAMA_URL)
logger.info("Qdrant Host: %s", settings.QDRANT_HOST)
