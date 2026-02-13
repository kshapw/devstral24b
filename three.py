from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
print(client.count(collection_name="ksk_docs"))
