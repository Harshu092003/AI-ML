import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.documents import Document

rag_chain = None
patient_map = {}   # for exact lookup


def initialize_rag():
    global rag_chain, patient_map

    persist_directory = "./chromadb"

    # Embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2"
    )

    # LLM
    llm = ChatOllama(
        model="llama3",
        temperature=0
    )

    # Load or create DB
    if os.path.exists(persist_directory):
        print("Loading existing vector database")

        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

    else:
        print("Creating new vector database")

        documents = PyPDFDirectoryLoader("pdf/").load()

        # combine all pages
        raw_text = "\n".join([doc.page_content for doc in documents])

        # STEP 2: Split by patient
        patients = raw_text.split("Patient Name:")

        clean_docs = []
        for p in patients:
            p = p.strip()
            if p:
                full_record = "Patient Name: " + p
                clean_docs.append(full_record)

        print("Total patients:", len(clean_docs))

        # NEW: Create mapping to preserve metadata (file + page)
        patient_metadata_map = {}

        for doc in documents:
            text = doc.page_content
            source_file = doc.metadata.get("source", "unknown")
            page_no = int(doc.metadata.get("page", 0))  #  ensure int

            parts = text.split("Patient Name:")
            for part in parts:
                part = part.strip()
                if not part:
                    continue

                full_record = "Patient Name: " + part
                key = full_record.split("\n")[0].strip()
                key = key.replace("  ", " ").strip()

                if key not in patient_metadata_map:
                    patient_metadata_map[key] = {
                        "pages": set(),   #  use set internally
                        "source": source_file
                    }

                patient_metadata_map[key]["pages"].add(page_no)

        # STEP 3: Convert to documents (NO CHUNKING)
        docs = []
        for p in clean_docs:
            key = p.split("\n")[0].strip()
            key = key.replace("  ", " ").strip()

            metadata = patient_metadata_map.get(key, {})

            pages = list(metadata.get("pages", []))

            #  store only ONE page in DB (important)
            page_to_store = pages[0] if pages else 0

            docs.append(
                Document(
                    page_content=p,
                    metadata={
                        "source": metadata.get("source", "unknown"),
                        "page": page_to_store   #  ONLY int allowed
                    }
                )
            )

        # STEP 4: Create vector DB
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory
        )

        vectordb.persist()

        # STEP 5: Build exact lookup map
        for doc in docs:
            key = doc.page_content.split("\n")[0].strip()
            patient_map[key] = doc.page_content

    # FIX: Use similarity (NOT MMR)
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # Prompt
    system_prompt = (
        "Use only the given context to answer the question. "
        "If the answer is not present, say 'I don't know'.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    print("RAG initialized successfully")


def ask_question(question: str):
    global rag_chain

    if rag_chain is None:
        raise RuntimeError("RAG not initialized")

    question = f"query: {question}"

    response = rag_chain.invoke({
        "input": question
    })

    answer = response["answer"]
    sources = response["context"]

    # Find BEST matching page
    best_page = None
    best_file = "unknown"
    max_match_score = 0

    for doc in sources:
        content = doc.page_content.lower()
        ans = answer.lower()

        # simple relevance score (word overlap)
        score = sum(1 for word in ans.split() if word in content)

        if score > max_match_score:
            max_match_score = score
            best_page = doc.metadata.get("page", None)
            best_file = doc.metadata.get("source", "unknown")

    return {
        "answer": answer,
        "sources": [
            {
                "file": best_file,
                "pages": [best_page] if best_page is not None else []
            }
        ]
    }
    
    

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