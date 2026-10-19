from pathlib import Path

# Folders
ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
DB_DIR = ROOT / "db_chromadb"
CHUNKS_FILE = ROOT / "chunks.json"   # a copy of all chunks, used later for keyword search

# Models (all free and local)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "llama3.2"

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150