# https://github.com/datalab-to/marker  use this for preprocessing

import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Docling for high-quality parsing
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions

# LlamaIndex core
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

load_dotenv()

# --------------------------------------------------
# 1. System Configuration
# --------------------------------------------------
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.llm = Ollama(model="llama3", request_timeout=300.0)
Settings.chunk_size = 1024 # Larger chunks help keep Table Row + Date together
Settings.chunk_overlap = 200

# --------------------------------------------------
# 2. Advanced OCR (Fixes "Empty Result" issues)
# --------------------------------------------------
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.ocr_options = RapidOcrOptions()
pipeline_options.ocr_options.force_full_page_ocr = True # Forces OCR even if text is "hidden"

converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

# --------------------------------------------------
# 3. Processing Documents
# --------------------------------------------------
documents = []
pdf_dir = Path("./pdf")

for pdf_file in pdf_dir.glob("*.pdf"):
    print(f"📄 Analyzing Report: {pdf_file.name}...")
    result = converter.convert(pdf_file)
    # Use markdown because it preserves table rows perfectly for the AI
    markdown_output = result.document.export_to_markdown()
    
    documents.append(
        Document(
            text=markdown_output,
            metadata={"filename": pdf_file.name}
        )
    )

# --------------------------------------------------
# 4. Fresh Database (Prevents data mixing from old runs)
# --------------------------------------------------
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection(name="medical_records")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=ChatMemoryBuffer.from_defaults(token_limit=4000),
    similarity_top_k=15, # Increased K to ensure we find ALL reports in the folder
)

# --------------------------------------------------
# 6. Interaction
# --------------------------------------------------
print("\n✅ System Ready. Ask about any patient or test.")

while True:
    user_input = input("\nDoctor: ")
    if user_input.lower() in ["exit", "quit"]: break

    response = chat_engine.chat(user_input)
    print("\nAI Assistant:", response.response)