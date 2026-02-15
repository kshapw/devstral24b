from app.ollama_client import OllamaClient, default_ollama
from app.config import settings
from qdrant_client import AsyncQdrantClient
from app.qdrant_service import get_qdrant_client

# We allow passing clients to support dependency injection from FastAPI lifespan
async def retrieve(query: str, qdrant: AsyncQdrantClient = None, ollama: OllamaClient = None, top_k: int = 5):
    qdrant = qdrant or get_qdrant_client()
    ollama = ollama or default_ollama

    vector = await ollama.embed(query)

    results = await qdrant.query_points(
        collection_name=settings.COLLECTION_NAME,
        query=vector,
        limit=top_k
    )
    
    # query_points returns a generic object, access points
    points = results.points

    context = "\n\n".join([r.payload["text"] for r in points])
    return context


async def answer(question: str, qdrant: AsyncQdrantClient = None, ollama: OllamaClient = None):
    # Pass clients down
    context = await retrieve(question, qdrant=qdrant, ollama=ollama)

    prompt = f"""
You are a support assistant for Karmika Seva Kendra.

Answer only from the context.
If information is missing, say you don't know.

Context:
{context}

Question:
{question}

Answer:
"""
    ollama = ollama or default_ollama
    return await ollama.generate(prompt)
