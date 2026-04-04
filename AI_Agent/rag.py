"""
rag.py
──────
Handles:
  • Loading PDFs and splitting them patient-by-patient
  • Building / loading the ChromaDB vector store
  • Retrieving relevant context for a query
  • Calling Ollama (LLaMA3) to generate an answer
"""

import os
import re
import json
import requests
from typing  import Generator

from langchain_community.embeddings    import HuggingFaceEmbeddings
from langchain_community.vectorstores  import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents          import Document

from config  import (OLLAMA_BASE_URL, OLLAMA_MODEL, EMBEDDING_MODEL,
                     PDF_DIR, CHROMA_DIR, RETRIEVER_K)
from logger  import log


# ── Module-level retriever (initialised once) ─────────────────
_retriever = None


# ─────────────────────────────────────────────────────────────
# 1. INITIALISE RAG
# ─────────────────────────────────────────────────────────────
def initialize_rag() -> None:
    """Load or build the vector store, set the global retriever."""
    global _retriever

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        log.info("Loading existing ChromaDB vector store...")
        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
    else:
        log.info("Building new ChromaDB vector store from PDFs...")
        vectordb = _build_vectorstore(embeddings)

    _retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K}
    )
    log.info("RAG initialised successfully ✅")


def _build_vectorstore(embeddings) -> Chroma:
    """Parse PDFs patient-by-patient and persist to ChromaDB."""

    raw_documents = PyPDFDirectoryLoader(PDF_DIR).load()
    if not raw_documents:
        raise FileNotFoundError(f"No PDFs found in '{PDF_DIR}'. Add PDFs and retry.")

    raw_text = "\n".join(doc.page_content for doc in raw_documents)

    # ── Build patient → metadata map ──────────────────────────
    patient_meta: dict = {}
    for doc in raw_documents:
        source = doc.metadata.get("source", "unknown")
        page   = int(doc.metadata.get("page", 0))
        for part in doc.page_content.split("Patient Name:"):
            part = part.strip()
            if not part:
                continue
            key = "Patient Name: " + part.split("\n")[0].strip()
            if key not in patient_meta:
                patient_meta[key] = {"pages": set(), "source": source}
            patient_meta[key]["pages"].add(page)

    # ── Build one Document per patient ────────────────────────
    docs = []
    for chunk in raw_text.split("Patient Name:"):
        chunk = chunk.strip()
        if not chunk:
            continue
        full_text = "Patient Name: " + chunk
        key       = full_text.split("\n")[0].strip()
        meta      = patient_meta.get(key, {})
        pages     = sorted(meta.get("pages", [0]))
        docs.append(Document(
            page_content=full_text,
            metadata={
                "source": meta.get("source", "unknown"),
                "page"  : pages[0]
            }
        ))

    log.info(f"Indexing {len(docs)} patient records...")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vectordb.persist()
    log.info("Vector store saved to disk ✅")
    return vectordb


# ─────────────────────────────────────────────────────────────
# 2. RETRIEVE CONTEXT
# ─────────────────────────────────────────────────────────────
def get_context_and_source(question: str) -> tuple[str, dict]:
    """
    Run similarity search, return:
      • context  : concatenated patient text for the LLM prompt
      • source   : {"file": ..., "page": ...} of best-matching doc
    """
    if _retriever is None:
        raise RuntimeError("RAG not initialised — call initialize_rag() first.")

    docs    = _retriever.invoke(f"query: {question}")
    context = "\n\n".join(doc.page_content for doc in docs)

    # Simple re-rank: pick doc with most word overlap to question
    q_words  = set(question.lower().split())
    best_doc = max(
        docs,
        key=lambda d: sum(1 for w in q_words if w in d.page_content.lower()),
        default=docs[0]
    )
    source = {
        "file": best_doc.metadata.get("source", "unknown"),
        "page": best_doc.metadata.get("page",   0)
    }
    return context, source


# ─────────────────────────────────────────────────────────────
# 3. GENERATE ANSWER  (non-streaming, for agent / email)
# ─────────────────────────────────────────────────────────────
def generate_answer(question: str, context: str) -> str:
    """Call Ollama and return the complete answer string."""
    payload = {
        "model" : OLLAMA_MODEL,
        "prompt": (
            "You are a medical assistant. "
            "Use ONLY the context below to answer the question. "
            "If the answer is not in the context, say 'I don't know'.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        ),
        "stream" : False,
        "options": {"temperature": 0, "num_predict": 512}
    }
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=120
    )
    resp.raise_for_status()
    return resp.json().get("response", "No answer generated.").strip()


# ─────────────────────────────────────────────────────────────
# 4. STREAM ANSWER  (for terminal / UI)
# ─────────────────────────────────────────────────────────────
def stream_answer(question: str) -> Generator[str, None, None]:
    """
    Yield answer tokens one by one.
    Last yielded item is the [SOURCE] citation line.
    """
    context, source = get_context_and_source(question)

    payload = {
        "model" : OLLAMA_MODEL,
        "prompt": (
            "You are a medical assistant. "
            "Use ONLY the context below to answer the question. "
            "If the answer is not in the context, say 'I don't know'.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        ),
        "stream" : True,
        "options": {"temperature": 0, "num_predict": 512}
    }

    with requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        stream=True,
        timeout=120
    ) as resp:
        for line in resp.iter_lines():
            if line:
                data  = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done", False):
                    yield f"\n\n[SOURCE] {source['file']} | Page {source['page'] + 1}"
                    break


# ─────────────────────────────────────────────────────────────
# 5. HELPER — extract patient name from context
# ─────────────────────────────────────────────────────────────
def extract_patient_name(context: str) -> str:
    match = re.search(r"Patient Name:\s*(.+)", context)
    return match.group(1).strip() if match else "Unknown Patient"