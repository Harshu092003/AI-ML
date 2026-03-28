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
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
import chromadb
import os

# Load environment variables
load_dotenv()

# Global query engine
query_engine = None


def initialize_rag():
    global query_engine

    # Default chunk settings (less relevant now, but kept)
    Settings.chunk_size = 600
    Settings.chunk_overlap = 200

    # Embedding model (used for semantic search)
    embed_model = HuggingFaceEmbedding(
        model_name="intfloat/e5-base-v2",
        embed_batch_size=32,
    )
    Settings.embed_model = embed_model

    # LLM (used to generate final answers)
    Settings.llm = Ollama(
        model="llama3",
        request_timeout=120.0
    )

    # Initialize ChromaDB (must be BEFORE load/create index)
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

    # Check if data already exists in DB
    existing_data = chroma_collection.get()

    if existing_data and len(existing_data["ids"]) > 0:
        print("Loading existing index...")

        # Load index without reprocessing PDFs
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store
        )

    else:
        print("Creating new index...")

        # Load documents
        documents = SimpleDirectoryReader("pdf/").load_data()

        # Semantic splitter (meaning-based chunking)
        semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=embed_model,
        )

        # Fallback splitter (for large chunks)
        sentence_splitter = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=100
        )

        # Step 1: Semantic chunking
        nodes = semantic_splitter.get_nodes_from_documents(documents)

        # Step 2: Recursive refinement
        refined_nodes = []
        for node in nodes:
            if len(node.text) > 800:
                refined_nodes.extend(
                    sentence_splitter.get_nodes_from_documents([node])
                )
            else:
                refined_nodes.append(node)

        # Create index and store embeddings
        index = VectorStoreIndex(
            refined_nodes,
            storage_context=storage_context
        )

    # Create query engine (used for asking questions)
    query_engine = index.as_query_engine(
        similarity_top_k=15
    )

    print("RAG initialized successfully")


def ask_question(question: str) -> str:
    if query_engine is None:
        raise RuntimeError("RAG engine not initialized")

    response = query_engine.query(question)
    return str(response)