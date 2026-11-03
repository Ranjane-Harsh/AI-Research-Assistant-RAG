# config.py
import os
from dotenv import load_dotenv

# Load variables from .env file into the environment
load_dotenv()

# --- File Paths ---
PAPERS_DIR = "data/papers"
CHROMA_DIR = "data/chroma_db"

# --- Embedding Model ---
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DEVICE = "cpu"    # Switch to "cpu" if CUDA unavailable

# --- Re-ranker Model (Stage 2: Cross-Encoder) ---
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANKER_DEVICE = "cpu"    # switch to "cpu" if CUDA unavailable

# --- LLM via Groq ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "openai/gpt-oss-120b"

# --- Chunking ---
CHUNK_SIZE    = 1500    # max characters per chunk
CHUNK_OVERLAP = 200     # characters shared between adjacent chunks

# --- Two-Stage Retrieval Parameters ---
INITIAL_TOP_K = 15          # Stage 1: ChromaDB retrieves top 15 broad candidates
FINAL_TOP_K   = 3           # Stage 2: Re-ranker filters down to top 3 best chunks