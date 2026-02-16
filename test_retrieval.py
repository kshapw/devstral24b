import asyncio
import logging

from qdrant_client import AsyncQdrantClient

from app.config import settings
from app.ollama_client import default_ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    logger.info(
        "Connecting to Qdrant at %s:%s", settings.QDRANT_HOST, settings.QDRANT_PORT
    )

    client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

    query = "accident death compensation amount"
    logger.info("Query: %s", query)

    try:
        vector = await default_ollama.embed(query)
        logger.info("Embedding length: %d", len(vector))

        results = await client.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=vector,
            limit=3,
        )

        points = results.points
        logger.info("Retrieved %d chunks", len(points))

        for i, p in enumerate(points, 1):
            logger.info("--- Result %d ---\n%s", i, p.payload["text"])
    finally:
        await client.close()
        await default_ollama.client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
