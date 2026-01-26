# provides chat like behaviour , stores previous conversation in memory for providing that , just similar like chat functions .


from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

load_dotenv()

# 🔹 Better chunking so only education is retrieved
Settings.chunk_size = 300
Settings.chunk_overlap = 50

# 🔹 Local embeddings (no OpenAI)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 🔹 Local LLM (llama3 via Ollama)
Settings.llm = Ollama(model="llama3")

# Load documents
documents = SimpleDirectoryReader("pdf/").load_data()

# Build index
index = VectorStoreIndex.from_documents(documents)

chat_engine = index.as_chat_engine(chat_mode="best",llm = Settings.llm , verbose = True)
response = chat_engine.chat("what issue ayesha qureshi haves ?")

print(response)