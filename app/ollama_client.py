import requests
from app.config import settings

def generate(prompt):
    response = requests.post(
        f"{settings.OLLAMA_URL}/api/generate",
        json={
            "model": settings.LLM_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]


def embed(text):
    response = requests.post(
        f"{settings.OLLAMA_URL}/api/embeddings",
        json={
            "model": settings.EMBED_MODEL,
            "prompt": text
        }
    )
    return response.json()["embedding"]
