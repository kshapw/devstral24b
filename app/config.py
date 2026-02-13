import os

class Settings:
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
    LLM_MODEL = os.getenv("LLM_MODEL", "devstral:24b")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

    QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
    QDRANT_PORT = 6333
    COLLECTION_NAME = "ksk_docs"
    VECTOR_SIZE = 768

settings = Settings()
