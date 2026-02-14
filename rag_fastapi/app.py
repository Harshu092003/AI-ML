# To run : uvicorn app:app --reload

from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from rag_engine import initialize_rag, ask_question

app = FastAPI(title="Simple RAG API")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(" Initializing RAG...")
    initialize_rag()
    print(" RAG ready")
    yield
    

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    answer = ask_question(request.question)
    return {"answer": answer}
