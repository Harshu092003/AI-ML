import os
import requests
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document

retriever_global = None

def initialize_rag():
    global retriever_global

    persist_directory = "./chromadb"
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")

    if os.path.exists(persist_directory):
        print("Loading existing vector database")
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    else:
        print("Creating new vector database")
        documents = PyPDFDirectoryLoader("pdf/").load()
        raw_text = "\n".join([doc.page_content for doc in documents])

        # Build patient-wise metadata map
        patient_metadata_map = {}
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            page = int(doc.metadata.get("page", 0))
            for part in doc.page_content.split("Patient Name:"):
                part = part.strip()
                if not part:
                    continue
                key = "Patient Name: " + part.split("\n")[0].strip()
                if key not in patient_metadata_map:
                    patient_metadata_map[key] = {"pages": set(), "source": source}
                patient_metadata_map[key]["pages"].add(page)

        # Build docs
        docs = []
        for p in raw_text.split("Patient Name:"):
            p = p.strip()
            if not p:
                continue
            full = "Patient Name: " + p
            key = full.split("\n")[0].strip()
            meta = patient_metadata_map.get(key, {})
            pages = list(meta.get("pages", []))
            docs.append(Document(
                page_content=full,
                metadata={"source": meta.get("source", "unknown"), "page": pages[0] if pages else 0}
            ))

        vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
        vectordb.persist()

    retriever_global = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    print("RAG initialized successfully")


def get_context_and_sources(question: str):
    """Retrieve docs and return context string + best source."""
    docs = retriever_global.invoke(f"query: {question}")
    context = "\n\n".join([doc.page_content for doc in docs])

    # Pick best source doc by word overlap (simple re-ranking)
    best_doc = max(docs, key=lambda d: sum(
        1 for w in question.lower().split() if w in d.page_content.lower()
    ), default=docs[0])

    source = {
        "file": best_doc.metadata.get("source", "unknown"),
        "page": best_doc.metadata.get("page", 0)
    }
    return context, source


def stream_question(question: str):
    """Yields tokens, then a final JSON citation line."""
    if retriever_global is None:
        raise RuntimeError("RAG not initialized")

    context, source = get_context_and_sources(question)

    payload = {
        "model": "llama3",
        "prompt": (
            "Use only the given context to answer the question. "
            "If the answer is not present, say 'I don't know'.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        ),
        "stream": True,
        "options": {"temperature": 0, "num_predict": 512}
    }

    with requests.post("http://localhost:11434/api/generate", json=payload, stream=True) as resp:
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done", False):
                    # Send citation as a final special line
                    yield f"\n\n[SOURCE] {source['file']} | Page {source['page'] + 1}"
                    break
# 🔹 Techniques Used in This RAG Pipeline
# 1. Retrieval-Augmented Generation (RAG)
# 2. Dense Vector Search (Embedding-based retrieval using E5 model)
# 3. Semantic Search (via HuggingFace embeddings)
# 4. Vector Database (ChromaDB for storage & retrieval)

# 5. Custom Document Structuring (Patient-wise splitting instead of page-wise)
# 6. Metadata Mapping (Tracking source file + page number)
# 7. Deduplication (Using set() for unique page numbers)

# 8. Top-K Retrieval (k=3 optimization)
# re-ranking with simple word-overlap scoring to find best source page for answer
# 9. Context Window Optimization (Reducing unnecessary documents)

# 10. Relevance Scoring (Word-overlap based scoring)
# 11. Basic Re-ranking (Selecting best document using score)

# 12. Source Attribution (Returning file name + exact page)
# 13. Query Formatting (E5 model prefix: "query:")

# 14. Stuff Document Chain (Combining retrieved docs into single context)
# 15. Similarity Search (instead of MMR)

# 16. Exact Matching Optimization (patient_map for direct lookup - optional fast path)

# 17. Prompt Engineering (Strict instruction: "Use only given context")

# 18. Lightweight Context Filtering (Choosing best doc for final answer source)
# source citation with page number (not just file) for better traceability