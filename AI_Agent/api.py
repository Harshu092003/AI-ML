"""
api.py
──────
FastAPI server — exposes the agent as an HTTP API.

Start with:
    uvicorn api:app --reload --port 8000

Endpoints:
    POST /query   → run agent, optionally email result
    GET  /health  → health check
"""

from contextlib       import asynccontextmanager
from fastapi          import FastAPI, HTTPException
from pydantic         import BaseModel
from rag              import initialize_rag, stream_answer
from agent            import run_agent
from fastapi.responses import StreamingResponse


# ── Startup: initialise RAG once ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_rag()
    yield

app = FastAPI(
    title      = "Patient RAG Agent API",
    description= "Query patient records and email results via LLaMA3 + ChromaDB",
    version    = "1.0.0",
    lifespan   = lifespan
)


# ── Request / Response schemas ────────────────────────────────
class QueryRequest(BaseModel):
    question        : str
    recipient_email : str
    auto_send       : bool = True
    use_llm_decision: bool = False   # set True to let LLM decide whether to email

class QueryResponse(BaseModel):
    patient_name : str
    question     : str
    answer       : str
    source_file  : str
    source_page  : int
    emailed      : bool


# ── POST /query ───────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query_patient(req: QueryRequest):
    """
    Run full agent pipeline.
    Body example:
    {
        "question"        : "What is the diagnosis of Patient 132?",
        "recipient_email" : "doctor@hospital.com",
        "auto_send"       : true
    }
    """
    try:
        result = run_agent(
            question         = req.question,
            recipient_email  = req.recipient_email,
            auto_send        = req.auto_send,
            use_llm_decision = req.use_llm_decision
        )
        return QueryResponse(
            patient_name = result.patient_name,
            question     = result.question,
            answer       = result.answer,
            source_file  = result.source["file"],
            source_page  = result.source["page"] + 1,
            emailed      = result.emailed
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /stream?question=... ──────────────────────────────────
@app.get("/stream")
async def stream_patient_query(question: str):
    """
    Stream answer tokens in real time (Server-Sent Events style).
    Usage:  GET /stream?question=What+is+Patient+132+diagnosed+with
    """
    return StreamingResponse(
        stream_answer(question),
        media_type="text/plain"
    )


# ── GET /health ───────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model": "llama3"}