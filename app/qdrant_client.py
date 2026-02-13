from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from config import settings

client = QdrantClient(
    host=settings.QDRANT_HOST,
    port=settings.QDRANT_PORT
)

def create_collection():
    client.recreate_collection(
        collection_name=settings.COLLECTION_NAME,
        vectors_config=VectorParams(
            size=settings.VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )
