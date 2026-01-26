import streamlit as st
from dotenv import load_dotenv
import chromadb

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

# -------------------- One-time initialization --------------------
if "initialized" not in st.session_state:
    # Chunking
    Settings.chunk_size = 300
    Settings.chunk_overlap = 50

    # 🔹 Stable local embeddings
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        embed_batch_size=1,
    )

    # 🔹 Local LLM
    Settings.llm = Ollama(model="llama3")

    # 🔹 ChromaDB (persistent)
    chroma_client = chromadb.Client(
        settings=chromadb.Settings(
            persist_directory="./chroma_db"
        )
    )

    chroma_collection = chroma_client.get_or_create_collection(
        name="medical_reports"
    )

    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    # Load documents
    documents = SimpleDirectoryReader("pdf/").load_data()

    # Build index (uses ChromaDB)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    # Save in session
    st.session_state.index = index
    st.session_state.initialized = True

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Savvy AI", layout="centered")
st.title("Savvy AI ")

query = st.text_input(
    "Ask a question",
    placeholder="Voriconazole test report of Ayesha"
)

if st.button("Get Answer"):
    if query.strip():
        with st.spinner("Thinking..."):
            query_engine = st.session_state.index.as_query_engine(
                similarity_top_k=2,
                response_mode="compact",
            )
            response = query_engine.query(query)

        st.subheader("Answer")
        st.write(response.response)
    else:
        st.warning("Please enter a question.")


# to run
# streamlit run rag.py