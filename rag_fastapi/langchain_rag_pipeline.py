

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
rag_chain = None 

def initialize_rag():
    global rag_chain

    persist_directory = "./chroma_db"

    # Step 1: Embedding model (converts text -> vectors)
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2"
    )

    # Step 2: LLM (local model using Ollama)
    llm = ChatOllama(
        model="llama3",
        temperature=0
    )

    # Step 3: Load or create vector database
    if os.path.exists(persist_directory):
        print("Loading existing vector database...")

        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

    else:
        print("Creating new vector database...")

        # Load PDFs
        documents = PyPDFDirectoryLoader("pdf/").load()

        # Split into smaller chunks (important for accuracy)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=200
        )

        docs = text_splitter.split_documents(documents)

        # Create vector DB
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory
        )

        vectordb.persist()

    # Step 4: Retriever (fetch top relevant chunks)
    retriever = vectordb.as_retriever(
        search_type="mmr" , # its an hybrid search that combines relevance + diversity, good for long docs 
        search_kwargs={"k": 5}
    )

    # Step 5: Prompt (controls LLM behavior)
    system_prompt = (
        "Use ONLY the given context to answer the question. "
        "If answer is not present, say 'I don't know'.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # Step 6: Combine documents + LLM
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Step 7: Create final RAG chain
    rag_chain = create_retrieval_chain(retriever, document_chain)

    print("RAG initialized successfully")


def ask_question(question: str) -> str:
    global rag_chain

    if rag_chain is None:
        raise RuntimeError("RAG not initialized")

    # E5 model requires query prefix
    question = f"query: {question}"

    # Run pipeline
    response = rag_chain.invoke({
        "input": question
    })

    # Final answer
    return response["answer"]