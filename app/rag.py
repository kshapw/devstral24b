from ollama_client import embed, generate
from qdrant_client import client
from config import settings

def retrieve(query, top_k=5):
    vector = embed(query)

    results = client.search(
        collection_name=settings.COLLECTION_NAME,
        query_vector=vector,
        limit=top_k
    )

    context = "\n\n".join([r.payload["text"] for r in results])
    return context


def answer(question):
    context = retrieve(question)

    prompt = f"""
You are a support assistant for Karmika Seva Kendra.

Answer only from the context.
If information is missing, say you don't know.

Context:
{context}

Question:
{question}

Answer:
"""

    return generate(prompt)
