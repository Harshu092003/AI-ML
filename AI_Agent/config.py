"""
config.py
─────────
Single source of truth for all settings.
Reads from .env file — never hardcode secrets in other files.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env from project root

# ── Ollama / LLM ─────────────────────────────────────────────
OLLAMA_BASE_URL : str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    : str = os.getenv("OLLAMA_MODEL",    "llama3")

# ── Embeddings ───────────────────────────────────────────────
EMBEDDING_MODEL : str = os.getenv("EMBEDDING_MODEL", "intfloat/e5-base-v2")

# ── Paths ────────────────────────────────────────────────────
PDF_DIR    : str = os.getenv("PDF_DIR",    "./pdf")
CHROMA_DIR : str = os.getenv("CHROMA_DIR", "./chromadb")
LOG_DIR    : str = os.getenv("LOG_DIR",    "./logs")

# ── Gmail ────────────────────────────────────────────────────
SENDER_EMAIL      : str = os.getenv("SENDER_EMAIL",       "")
GMAIL_APP_PASSWORD: str = os.getenv("GMAIL_APP_PASSWORD", "")
SMTP_SERVER       : str = "smtp.gmail.com"
SMTP_PORT         : int = 587

# ── RAG retriever ────────────────────────────────────────────
RETRIEVER_K       : int = 5   # how many chunks to retrieve per query