import logging

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import settings

logger = logging.getLogger(__name__)


def get_qdrant_client() -> AsyncQdrantClient:
    logger.info(
        "Creating Qdrant client for %s:%s", settings.QDRANT_HOST, settings.QDRANT_PORT
    )
    return AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)


async def create_collection(client: AsyncQdrantClient) -> None:
    try:
        exists = await client.collection_exists(settings.COLLECTION_NAME)
    except Exception:
        logger.error(
            "Failed to check existence of Qdrant collection '%s'",
            settings.COLLECTION_NAME, exc_info=True,
        )
        raise

    if not exists:
        logger.info(
            "Creating collection '%s' (size=%d, distance=COSINE)",
            settings.COLLECTION_NAME,
            settings.VECTOR_SIZE,
        )
        try:
            await client.create_collection(
                collection_name=settings.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=settings.VECTOR_SIZE, distance=Distance.COSINE
                ),
            )
        except Exception:
            logger.error(
                "Failed to create Qdrant collection '%s'",
                settings.COLLECTION_NAME, exc_info=True,
            )
            raise
    else:
        logger.info("Collection '%s' already exists", settings.COLLECTION_NAME)
