import json
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src import config


def load_pdfs():
    """Read every PDF in docs/ and return one item per page."""
    pdf_files = sorted(config.DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"No PDFs found in {config.DOCS_DIR}")

    pages = []
    for pdf in pdf_files:
        loaded = PyPDFLoader(str(pdf)).load()
        count = 0
        for p in loaded:
            if not p.page_content.strip():
                continue   # skip empty pages (for example scanned images)
            # Keep only the labels we need. page + 1 gives human page numbers.
            pages.append(Document(
                page_content=p.page_content,
                metadata={"doc": pdf.name, "page": p.metadata.get("page", 0) + 1},
            ))
            count += 1
        print(f"  {pdf.name}: {count} pages with text (of {len(loaded)})")
    return pages


def build_chunks(pages):
    """Cut pages into smaller chunks and give each chunk a unique id."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)

    counters = {}
    for c in chunks:
        key = (c.metadata["doc"], c.metadata["page"])
        counters[key] = counters.get(key, 0) + 1
        c.metadata["chunk_id"] = f"{key[0]}|p{key[1]}|c{counters[key]}"
    return chunks


def save_chunks_json(chunks):
    """Save a plain copy of the chunks. Keyword search will use it later."""
    data = [
        {
            "id": c.metadata["chunk_id"],
            "doc": c.metadata["doc"],
            "page": c.metadata["page"],
            "text": c.page_content,
        }
        for c in chunks
    ]
    config.CHUNKS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=1))


def store_in_database(chunks):
    """Turn chunks into embeddings and store them in ChromaDB."""
    if config.DB_DIR.exists():
        shutil.rmtree(config.DB_DIR)   # start clean so nothing is stored twice

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        ids=[c.metadata["chunk_id"] for c in chunks],
        persist_directory=str(config.DB_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def main():
    print("Reading PDFs...")
    pages = load_pdfs()

    chunks = build_chunks(pages)
    print(f"\nTotal: {len(pages)} pages -> {len(chunks)} chunks")

    per_doc = {}
    for c in chunks:
        per_doc[c.metadata["doc"]] = per_doc.get(c.metadata["doc"], 0) + 1
    for name, n in per_doc.items():
        print(f"  {name}: {n} chunks")

    save_chunks_json(chunks)
    print(f"\nSaved chunk copy to {config.CHUNKS_FILE.name}")

    print("Creating embeddings and storing them (first time can take a minute)...")
    store_in_database(chunks)
    print(f"Done. Database saved in {config.DB_DIR.name}/")


if __name__ == "__main__":
    main()