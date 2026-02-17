import asyncio
import logging
import uuid

from qdrant_client.models import PointStruct

from app.chunker import chunk_markdown
from app.config import settings
from app.ollama_client import default_ollama
from app.qdrant_service import create_collection, get_qdrant_client

logger = logging.getLogger(__name__)


async def _embed_chunk(
    index: int,
    chunk: str,
    semaphore: asyncio.Semaphore,
) -> PointStruct | None:
    """Embed a single chunk, respecting the concurrency semaphore."""
    async with semaphore:
        try:
            vector = await default_ollama.embed(chunk)
            return PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": chunk},
            )
        except Exception:
            logger.error("Failed to embed chunk %d, skipping", index, exc_info=True)
            return None


async def ingest() -> None:
    data_path = settings.DATA_PATH
    logger.info("Reading data from %s", data_path)

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        logger.error("Data file not found: %s", data_path)
        raise
    except Exception:
        logger.error("Failed to read data file: %s", data_path, exc_info=True)
        raise

    chunks = chunk_markdown(content)
    logger.info("Produced %d chunks from input file", len(chunks))

    client = get_qdrant_client()
    try:
        # Recreate collection to remove old chunks
        if await client.collection_exists(settings.COLLECTION_NAME):
            logger.info("Deleting existing collection '%s'", settings.COLLECTION_NAME)
            await client.delete_collection(settings.COLLECTION_NAME)

        await create_collection(client)

        # Embed all chunks concurrently with bounded parallelism
        semaphore = asyncio.Semaphore(settings.INGEST_CONCURRENCY)
        tasks = [
            _embed_chunk(i, chunk, semaphore) for i, chunk in enumerate(chunks)
        ]
        results = await asyncio.gather(*tasks)

        # Filter out failed embeddings (None values)
        points = [r for r in results if r is not None]
        failed_count = len(results) - len(points)
        if failed_count > 0:
            logger.warning(
                "%d of %d chunks failed to embed and were skipped",
                failed_count, len(results),
            )

        if not points:
            logger.warning("No chunks were embedded successfully; nothing to ingest")
            return

        await client.upsert(
            collection_name=settings.COLLECTION_NAME,
            points=points,
        )

        logger.info(
            "Ingested %d chunks into collection '%s'",
            len(points),
            settings.COLLECTION_NAME,
        )
    finally:
        await client.close()
        await default_ollama.client.aclose()


if __name__ == "__main__":
    asyncio.run(ingest())
