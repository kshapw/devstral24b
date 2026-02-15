import time
import asyncio
import httpx
from datetime import datetime

from app.rag import answer
from app.config import settings
from app.ollama_client import default_ollama
from app.qdrant_service import get_qdrant_client

# ----------------------------
# Warmup (important for Devstral)
# ----------------------------
async def warmup_model():
    print("\nWarming up model...")
    try:
        # We can use default_ollama.generate if we want, or raw httpx
        # Let's use the client method to test the path
        await default_ollama.generate("warmup")
        print("Warmup complete.\n")
    except Exception as e:
        print("Warmup failed:", e)
        print("Continuing without warmup...\n")


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
    "Where can I apply for Karmika schemes?"
]

output_file = "rag_performance.txt"


# ----------------------------
# Benchmark
# ----------------------------
async def run_benchmark():
    # Setup clients for the script
    qdrant = get_qdrant_client()
    
    # We will pass these explicit clients to answer to test the injection behavior primarily
    # though answer() defaults to global if None, let's be explicit to mimic main.py
    
    print("\nRunning RAG performance test...\n")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"RAG Performance Test - {datetime.now()}\n")
        f.write(f"Model: {settings.LLM_MODEL}\n")
        f.write(f"Ollama: {settings.OLLAMA_URL}\n")
        f.write("=" * 60 + "\n\n")

        times = []

        try:
            for i, question in enumerate(queries, 1):
                print(f"Query {i}: {question}")

                start_time = time.time()

                # Passing dependencies
                response = await answer(question, qdrant=qdrant, ollama=default_ollama)

                end_time = time.time()
                duration = round(end_time - start_time, 2)
                times.append(duration)

                log = (
                    f"Query {i}\n"
                    f"Question: {question}\n"
                    f"Answer: {response}\n"
                    f"Turnaround Time: {duration} seconds\n"
                    f"{'-'*60}\n"
                )

                f.write(log)
                print(f"Completed in {duration} sec\n")

            # Summary
            avg_time = round(sum(times) / len(times), 2)
            f.write("\nSUMMARY\n")
            f.write(f"Average Response Time: {avg_time} seconds\n")

            print(f"\nAverage Response Time: {avg_time} sec")
            print(f"Results saved to {output_file}")
            
        finally:
            await qdrant.close()
            await default_ollama.client.aclose()


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    async def main():
        await warmup_model()
        await run_benchmark()
    
    asyncio.run(main())
