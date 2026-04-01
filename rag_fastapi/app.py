# cd into the rag_fastapi directory and run:
# uvicorn app:app --reload

# for testing the streaming endpoint, you can use curl:
# curl --no-buffer "http://localhost:8000/stream?question=Patient%20200"



import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse
from langchain_rag_pipeline import initialize_rag, ask_question, stream_question

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing RAG...")
    initialize_rag()
    print("RAG ready")
    yield

app = FastAPI(title="Simple RAG API", lifespan=lifespan)

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

async def stream_answer(question: str):
    for chunk in stream_question(question):
        yield chunk
        await asyncio.sleep(0)

@app.get("/stream")
async def stream_rag(question: str):
    return StreamingResponse(
        stream_answer(question),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        }
    )