# query engine is used for one question one answer concept and it doesnt store anything from previously asked questions.


from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

load_dotenv()

Settings.chunk_size = 300
Settings.chunk_overlap = 50

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

Settings.llm = Ollama(model="llama3")

documents = SimpleDirectoryReader("pdf/").load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(similarity_top_k=2)

response = query_engine.query("voriconazole test report of ayesha")
print(response)
