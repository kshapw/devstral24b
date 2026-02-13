import os

class Settings:
    # Environment mode: "host" or "docker"
    ENVIRONMENT = os.getenv("ENVIRONMENT", "host")

    # -------- Ollama --------
    if ENVIRONMENT == "docker":
        OLLAMA_URL = "http://ollama:11434"
    else:
        # host mode (port mapped: 11439 -> 11434)
        OLLAMA_URL = "http://localhost:11439"

    LLM_MODEL = os.getenv("LLM_MODEL", "devstral:24b")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

    # -------- Qdrant --------
    if ENVIRONMENT == "docker":
        QDRANT_HOST = "qdrant"
    else:
        QDRANT_HOST = "localhost"

    QDRANT_PORT = 6333
    COLLECTION_NAME = "ksk_docs"
    VECTOR_SIZE = 768


settings = Settings()

print(f"[CONFIG] Environment: {settings.ENVIRONMENT}")
print(f"[CONFIG] Ollama URL: {settings.OLLAMA_URL}")
print(f"[CONFIG] Qdrant Host: {settings.QDRANT_HOST}")

