import asyncio
import uuid
from app.chunker import chunk_markdown
from app.ollama_client import default_ollama
from app.config import settings
from app.qdrant_service import get_qdrant_client, create_collection

async def ingest():
    with open("data/ksk.md", "r", encoding="utf-8") as f:
        content = f.read()

    chunks = chunk_markdown(content)

    client = get_qdrant_client()
    try:
        # Recreate collection to remove old chunks
        if await client.collection_exists(settings.COLLECTION_NAME):
            await client.delete_collection(settings.COLLECTION_NAME)
            
        await create_collection(client)

        points = []
        for chunk in chunks:
            # default_ollama is a global instance we can use for scripts
            vector = await default_ollama.embed(chunk)

            points.append({
                "id": str(uuid.uuid4()),
                "vector": vector,
                "payload": {"text": chunk}
            })

        await client.upsert(
            collection_name=settings.COLLECTION_NAME,
            points=points
        )

        print(f"Ingested {len(points)} chunks")
    finally:
        await client.close()
        # default_ollama internal client might need closing if we were strict, 
        # but for a script it's fine. 
        # Ideally we'd do: await default_ollama.client.aclose()
        await default_ollama.client.aclose()


if __name__ == "__main__":
    asyncio.run(ingest())
