from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

#client = QdrantClient(host="qdrant", port=6333)
client = QdrantClient(host="localhost", port=6333)

client.recreate_collection(
    collection_name="ksk_docs",
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE
    )
)

print("Collection ready")
