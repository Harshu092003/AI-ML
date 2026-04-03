# cd into the rag_fastapi directory and run:
# uvicorn app:app --reload

# for testing the streaming endpoint, you can use curl:
# curl --no-buffer "http://localhost:8000/stream?question=Patient%20200"  %20 is used for spacing in the URL



import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse
from langchain_rag_pipeline import initialize_rag, stream_question

@asynccontextmanager # for shutdown and startup events in FastAPI
async def lifespan(app: FastAPI):
    print("Initializing RAG...")
    initialize_rag()
    print("RAG ready")
    yield

app = FastAPI(title="Simple RAG API", lifespan=lifespan)

class QueryRequest(BaseModel):
    question: str

async def stream_answer(question: str):
    for chunk in stream_question(question):
        yield chunk# yielding each token as it comes in, allowing for real-time streaming of the answer
        await asyncio.sleep(0) #asyncio sleep to allow other tasks to run while streaming

@app.get("/stream")
async def stream_rag(question: str):
    return StreamingResponse(
        stream_answer(question),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"} # browser will not store cache , and disable buffering for nginx if used as reverse proxy .
    )