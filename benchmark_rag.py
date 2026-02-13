import time
import requests
from datetime import datetime

from app.rag import answer
from app.config import settings


# ----------------------------
# Warmup (important for Devstral)
# ----------------------------
def warmup_model():
    print("\nWarming up model...")
    try:
        requests.post(
            f"{settings.OLLAMA_URL}/api/generate",
            json={
                "model": settings.LLM_MODEL,
                "prompt": "warmup",
                "stream": False
            },
            timeout=120
        )
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
def run_benchmark():
    print("\nRunning RAG performance test...\n")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"RAG Performance Test - {datetime.now()}\n")
        f.write(f"Model: {settings.LLM_MODEL}\n")
        f.write(f"Ollama: {settings.OLLAMA_URL}\n")
        f.write("=" * 60 + "\n\n")

        times = []

        for i, question in enumerate(queries, 1):
            print(f"Query {i}: {question}")

            start_time = time.time()

            response = answer(question)

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


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    warmup_model()
    run_benchmark()
