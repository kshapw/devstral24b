import uuid
from chunker import chunk_markdown
from app.ollama_client import embed
#from qdrant_client import client, create_collection
from app.config import settings
from qdrant_service import client, create_collection

def ingest():
    with open("data/ksk.md", "r", encoding="utf-8") as f:
        content = f.read()

    chunks = chunk_markdown(content)

    create_collection()

    points = []
    for chunk in chunks:
        vector = embed(chunk)

        points.append({
            "id": str(uuid.uuid4()),
            "vector": vector,
            "payload": {"text": chunk}
        })

    client.upsert(
        collection_name=settings.COLLECTION_NAME,
        points=points
    )

    print(f"Ingested {len(points)} chunks")


if __name__ == "__main__":
    ingest()
