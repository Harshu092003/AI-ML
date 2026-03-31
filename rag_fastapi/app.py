# change directory to cd rag_fastapi
# To run : uvicorn app:app --reload

from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langchain_rag_pipeline import initialize_rag, ask_question

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


class Source(BaseModel):
    file: str
    pages: list[int]


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    response = ask_question(request.question)
    return {
        "answer": response["answer"],
        "sources": response["sources"]
    }