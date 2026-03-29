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

        docs = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        ).split_documents(documents)

        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory
        )

        vectordb.persist()

    # Retriever (MMR = hybrid style)
    retriever = vectordb.as_retriever(
        search_type="mmr",
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

    # combine documents chain (stuffing) - simple approach where all retrieved documents are combined into a single prompt and sent to the LLM in one go. This method is straightforward but can be less efficient for large documents or when many documents are retrieved, as it may exceed token limits or lead to less focused responses.
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    print("RAG initialized successfully")


def ask_question(question: str) -> str:
    global rag_chain

    if rag_chain is None:
        raise RuntimeError("RAG not initialized")

    # E5 format
    question = f"query: {question}"

    response = rag_chain.invoke({
        "input": question
    })

    return response["answer"]