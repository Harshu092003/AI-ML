
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

load_dotenv()

query_engine = None


def initialize_rag():
    global query_engine

    
    Settings.chunk_size = 600
    Settings.chunk_overlap = 200

    
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="intfloat/e5-base-v2",
        embed_batch_size=32,
    )

    
    Settings.llm = Ollama(
        model="llama3",
        request_timeout=120.0
    )

    
    documents = SimpleDirectoryReader("pdf/").load_data()

    chroma_client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    chroma_collection = chroma_client.get_or_create_collection(
        name="medical_docs"
    )

    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )

    query_engine = index.as_query_engine(
        similarity_top_k=15
    )

    print(" RAG initialized using ChromaDB + E5")


def ask_question(question: str) -> str:
    if query_engine is None:
        raise RuntimeError("RAG engine not initialized")

    response = query_engine.query(question)
    return str(response)
