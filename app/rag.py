from app.ollama_client import embed, generate
#from qdrant_client import client
from app.config import settings
from app.qdrant_service import client

def retrieve(query, top_k=5):
    vector = embed(query)

    results = client.query_points(
        collection_name=settings.COLLECTION_NAME,
        query=vector,
        limit=top_k
    ).points

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
