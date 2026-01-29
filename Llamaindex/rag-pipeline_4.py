from dotenv import load_dotenv                 
import chromadb                              

from llama_index.core import (
    VectorStoreIndex,                         
    SimpleDirectoryReader,                   
    StorageContext,                           
    Settings,                                 
)
from llama_index.vector_stores.chroma import ChromaVectorStore  
from llama_index.core.memory import ChatMemoryBuffer             

from llama_index.embeddings.huggingface import HuggingFaceEmbedding  
from llama_index.llms.ollama import Ollama                          

load_dotenv()                               

Settings.chunk_size = 300                   
Settings.chunk_overlap = 50                 

Settings.embed_model = HuggingFaceEmbedding( 
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

Settings.llm = Ollama(model="llama3")        

documents = SimpleDirectoryReader("./pdf").load_data()  

db = chromadb.PersistentClient(path="./chroma_db")       # Persistent Chroma database
chroma_collection = db.get_or_create_collection(name="quickstart")  # Vector collection

vector_store = ChromaVectorStore(            # Wrap Chroma for LlamaIndex
    chroma_collection=chroma_collection
)

storage_context = StorageContext.from_defaults(  # Tell index where to store vectors
    vector_store=vector_store
)

index = VectorStoreIndex.from_documents(     # Create vector index from documents
    documents,
    storage_context=storage_context,
)

memory = ChatMemoryBuffer.from_defaults(     # Store chat history
    token_limit=3000
)

chat_engine = index.as_chat_engine(           # Create chat-based RAG engine
    chat_mode="context",                    
    memory=memory,                            # Enable conversation memory
    similarity_top_k=10,                       # Retrieve top 5 relevant chunks
)

while True:                                  
    user_input = input("\nYou: ")             
    if user_input.lower() in ["exit", "quit"]:  
        break

    response = chat_engine.chat(user_input)  
    print("\nAI:", response.response)         
