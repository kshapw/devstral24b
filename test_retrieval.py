from qdrant_client import QdrantClient
from app.ollama_client import embed
from app.config import settings

print("Connecting to Qdrant at:", settings.QDRANT_HOST, settings.QDRANT_PORT)

client = QdrantClient(
    host=settings.QDRANT_HOST,
    port=settings.QDRANT_PORT
)

query = "accident death compensation amount"
print("\nQuery:", query)

vector = embed(query)
print("Embedding length:", len(vector))

results = client.query_points(
    collection_name=settings.COLLECTION_NAME,
    query=vector,
    limit=3
)

points = results.points

print("\nRetrieved:", len(points), "chunks")

for i, p in enumerate(points, 1):
    print(f"\n--- Result {i} ---")
    print(p.payload["text"])
