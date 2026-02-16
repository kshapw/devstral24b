import asyncio
import logging

from app.rag import answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    question = "How much compensation is given if a worker dies in an accident?"
    logger.info("Question: %s", question)
    response = await answer(question)
    logger.info("Answer: %s", response)


if __name__ == "__main__":
    asyncio.run(main())
