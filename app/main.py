from fastapi import FastAPI
from pydantic import BaseModel
from rag import answer

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(q: Query):
    return {"answer": answer(q.question)}
