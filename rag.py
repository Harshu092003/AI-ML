
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
    embeddings = HuggingFaceEmbeddings(
        model_name = "intfloat/e5-base-v2"
    )

    llm = ChatOllama(model = "llama3" , temperature = 0)

    if os.path.exists(persist_directory):
        print("Loading existing vecotr database ")
        vectordb = Chroma(
            persist_directory = persist_directory ,
            embedding_function = embeddings
        )

    else :
        print("creating new vector database")
        documents = PyPDFDirectoryLoader("pdf/").load()

        docs = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""] 
        ).split_documents(documents)

        vectordb = Chroma.from_documents(
            documents = docs ,
            embeddings = embeddings ,
            persist_directory = persist_directory
        )

        vectordb.persist()
    
    retriever = vectordb.as_retriever(search_kwargs={"k": 3} , search_type = "mmr")
    system_prompt = (
        "Use only given context to answer the questions" ,
        "if answer not present say i dont know" ,
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate([
        ("system" , system_prompt) ,
        ("human" , "{input}")
    ])

    document_chain = create_stuff_documents_chain(llm,prompt)
    rag_chain = create_retrieval_chain(retriever,document_chain)

    print("Rag initialized successfully")

def ask_question(question : str) -> str :
    if rag_chain is None :
        raise RuntimeError("Rag not initialized")
    
    question = f"query: {question}"

    response = rag_chain.invoke({
        "input" : question ,
    })

    return response["answer"]







    