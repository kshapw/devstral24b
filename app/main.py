from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from app.rag import answer
from app.qdrant_service import get_qdrant_client
from app.ollama_client import OllamaClient
from qdrant_client import AsyncQdrantClient
import httpx

# Models
class Query(BaseModel):
    question: str

# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.qdrant = get_qdrant_client()
    app.state.http_client = httpx.AsyncClient(timeout=120.0)
    app.state.ollama = OllamaClient(client=app.state.http_client)
    
    yield
    
    # Shutdown
    await app.state.qdrant.close()
    await app.state.http_client.aclose()

app = FastAPI(lifespan=lifespan)

# Dependencies
def get_qdrant_dep():
    return app.state.qdrant

def get_ollama_dep():
    return app.state.ollama

@app.post("/chat")
async def chat(
    q: Query,
    qdrant: AsyncQdrantClient = Depends(get_qdrant_dep),
    ollama: OllamaClient = Depends(get_ollama_dep)
):
    result = await answer(q.question, qdrant=qdrant, ollama=ollama)
    return {"answer": result}
