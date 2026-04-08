

from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from langchain_rag import initialize_rag , stream_question , get_context_and_source
from fastapi.response import StreamingResponse

@asynccontextmanager
def lifespan(app:FastAPI) :
    print("initializing RAG")
    initialize_rag()
    print("RAG initialized")
    yield
    
app = FastAPI(Title = "RAG" , lifespan = lifespan)

class QueryQuestion(BaseModel):
    question : str
    
def stream_answer(question:str) :
    for token in stream_question(question):
        yield token
       
       
@app.get("/stream") 
def response(question : str):
    return StreamingResponse(
        stream_answer(question) , 
        media_type = "text/plain" , 
        headers = {"Cache-Control" : "no-cache" , "X-Accel-Buffering" : no}
        
    )
