import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

st.title("Chromadb + Hugging Face + Streamlit Demo ")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class HFEmbeddings:
    def __init__(self, model):
        self.model = model

    def __call__(self, input):
        return self.model.encode(input).tolist()

    def embed_query(self, input):
        return self.model.encode(input).tolist()

    def name(self):
        return "hf_embeddings"

embeddings = HFEmbeddings(model)

client = chromadb.Client(
    Settings(
        persist_directory="chroma_db",
        is_persistent=True
    )
)

collection = client.get_or_create_collection(
    name="my_documents",
    embedding_function=embeddings
)

import os

if len(collection.get()['ids']) == 0:
    if os.path.exists("documents.txt"):
        with open("documents.txt", "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        documents = lines
        metadatas = [{"source": f"doc{i+1}"} for i in range(len(documents))]
        ids = [f"doc{i+1}" for i in range(len(documents))]

        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        st.success(f"Added {len(documents)} documents to ChromaDB!")
    else:
        st.error("documents.txt file not found!")

# -----------------------------
query = st.text_input("Ask something:")

if query:
    results = collection.query(
        query_texts=[query],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    st.subheader("Top Matches:")
    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
        st.write(f"- {doc} (source: {metadata['source']})")

#test inputs 
# space
# vehicle
# energy
# science
# program
# programming