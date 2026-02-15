import httpx
from app.config import settings

class OllamaClient:
    def __init__(self, client: httpx.AsyncClient = None):
        self.client = client or httpx.AsyncClient(timeout=120.0)
        self.base_url = settings.OLLAMA_URL
        self.llm_model = settings.LLM_MODEL
        self.embed_model = settings.EMBED_MODEL

    async def generate(self, prompt: str) -> str:
        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["response"]
        except httpx.HTTPError as e:
            print(f"Error calling Ollama generate: {e}")
            raise

    async def embed(self, text: str) -> list[float]:
        try:
            response = await self.client.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.embed_model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except httpx.HTTPError as e:
            print(f"Error calling Ollama embed: {e}")
            raise

# Global/Singleton helper if needed, but preferably instantiated in lifespan
# We'll leave this here for simple script import, but intended for use with context manager
default_ollama = OllamaClient()
