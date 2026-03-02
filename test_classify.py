import asyncio
import os
from app.ollama_client import OllamaClient
from app.rag import _llm_classify_intent

async def main():
    ollama = OllamaClient("http://localhost:11441", "devstral:24b")
    intent = await _llm_classify_intent(ollama, "Renewal")
    print(f"Classification for 'Renewal': {intent}")
    intent2 = await _llm_classify_intent(ollama, "Registration")
    print(f"Classification for 'Registration': {intent2}")

if __name__ == "__main__":
    asyncio.run(main())
