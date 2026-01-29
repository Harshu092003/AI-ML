import os
from dotenv import load_dotenv
load_dotenv()

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor



import chromadb

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    load_index_from_storage,
)

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
resource = Resource.create(
    {
        # REQUIRED by Arize (at least one)
        "arize.project.name": "my-llamaindex-app",

        # STRONGLY recommended
        "model_id": "llama3-local-rag",

        # Optional but good practice
        "service.name": "llamaindex-rag",
    }
)

tracer_provider = TracerProvider(resource=resource)

exporter = OTLPSpanExporter(
    endpoint="https://otlp.arize.com/v1/traces",
    headers={
        "authorization": os.getenv("ARIZE_API_KEY"),
        "arize-space-id": os.getenv("ARIZE_SPACE_ID"),
    },
    timeout=30,
)

span_processor = BatchSpanProcessor(exporter)
tracer_provider.add_span_processor(span_processor)

trace.set_tracer_provider(tracer_provider)

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)


Settings.chunk_size = 300
Settings.chunk_overlap = 50

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

Settings.llm = Ollama(model="llama3")


documents = SimpleDirectoryReader("./pdf").load_data()

if os.path.exists("storage"):
    print("loading index from storage")

    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection(name="quickstart")

    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )

    storage_context = StorageContext.from_defaults(
        persist_dir="storage",
        vector_store=vector_store,
    )

    index = load_index_from_storage(storage_context)

else:
    print("creating index from documents")

    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection(name="quickstart")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    storage_context.persist(persist_dir="storage")

query_engine = index.as_query_engine()
response = query_engine.query("What is the document about?")
print(response)
