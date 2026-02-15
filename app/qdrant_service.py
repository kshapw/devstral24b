from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance
from app.config import settings

# We will instantiate this in the lifespan or use a singleton pattern that can be awaited
# For simplicity in this codebase, we can use a global variable but initialize it properly.
# However, purely async clients often need to be closed.
# Ideally, we return a client.

def get_qdrant_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT
    )

async def create_collection(client: AsyncQdrantClient):
    if not await client.collection_exists(settings.COLLECTION_NAME):
        await client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=settings.VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )

