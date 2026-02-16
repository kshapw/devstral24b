import asyncio
import logging
import time
from datetime import datetime

from app.config import settings
from app.ollama_client import default_ollama
from app.qdrant_service import get_qdrant_client
from app.rag import answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Warmup (important for Devstral)
# ----------------------------
async def warmup_model() -> None:
    logger.info("Warming up model...")
    try:
        await default_ollama.generate("warmup")
        logger.info("Warmup complete")
    except Exception as e:
        logger.warning("Warmup failed: %s", e)
        logger.info("Continuing without warmup")


# ----------------------------
# Test Queries
# ----------------------------
queries = [
    "How much compensation is given for accident death?",
    "What documents are required for accident assistance?",
    "Is FIR mandatory for accident claim?",
    "What is the pension amount after 60 years?",
    "How many times can I claim marriage assistance?",
    "What is the benefit under Thayi Magu scheme?",
    "How long do I have to apply after an accident?",
    "Can dependents apply for major illness treatment?",
    "What is the maximum amount for medical assistance?",
    "Where can I apply for Karmika schemes?",
]

output_file = "rag_performance.txt"


# ----------------------------
# Benchmark
# ----------------------------
async def run_benchmark() -> None:
    qdrant = get_qdrant_client()

    logger.info("Running RAG performance test")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"RAG Performance Test - {datetime.now()}\n")
        f.write(f"Model: {settings.LLM_MODEL}\n")
        f.write(f"Ollama: {settings.OLLAMA_URL}\n")
        f.write("=" * 60 + "\n\n")

        times: list[float] = []

        try:
            for i, question in enumerate(queries, 1):
                logger.info("Query %d: %s", i, question)

                start_time = time.time()

                response = await answer(
                    question, qdrant=qdrant, ollama=default_ollama
                )

                end_time = time.time()
                duration = round(end_time - start_time, 2)
                times.append(duration)

                log = (
                    f"Query {i}\n"
                    f"Question: {question}\n"
                    f"Answer: {response}\n"
                    f"Turnaround Time: {duration} seconds\n"
                    f"{'-' * 60}\n"
                )

                f.write(log)
                logger.info("Completed in %.2f sec", duration)

            avg_time = round(sum(times) / len(times), 2)
            f.write("\nSUMMARY\n")
            f.write(f"Average Response Time: {avg_time} seconds\n")

            logger.info("Average Response Time: %.2f sec", avg_time)
            logger.info("Results saved to %s", output_file)

        finally:
            await qdrant.close()
            await default_ollama.client.aclose()


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":

    async def main() -> None:
        await warmup_model()
        await run_benchmark()

    asyncio.run(main())
