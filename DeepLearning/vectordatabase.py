import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
from transformers import logging as hf_logging

hf_logging.set_verbosity_error() 

BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "chroma_db"

client = chromadb.Client(
    chromadb.config.Settings(
        persist_directory=str(PERSIST_DIR),
        is_persistent=True   
    )
)

collection = client.get_or_create_collection(name="docs")
print(f"Collection loaded. Current count: {collection.count()}")

model = SentenceTransformer("all-MiniLM-L6-v2")

if collection.count() == 0:
    documents = [
        "Apple is a fruit",
        "Car is a vehicle",
        "Banana is yellow",
        "Tesla makes electric cars"
    ]

    embeddings = model.encode(documents).tolist()

    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=[f"id_{i}" for i in range(len(documents))]
    )

    print("Documents inserted and persisted ")
else:
    print("Documents already exist ")

query = "Which things are vehicles?"
query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

print("\nQuery results:")
for doc in results["documents"][0]:
    print("-", doc)

print("\nPersistent directory:", PERSIST_DIR)
print("Exists on disk:", PERSIST_DIR.exists())
