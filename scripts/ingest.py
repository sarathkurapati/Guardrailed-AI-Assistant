"""
Ingest all .txt files from docs/sample_documents/ into ChromaDB.
Idempotent: skips if collection already has documents.

Usage: python scripts/ingest.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.vector_store import get_vector_store
from app.config import settings
from app.utils.logging import get_logger, configure_logging

configure_logging()
logger = get_logger(__name__)


def ingest():
    store = get_vector_store()

    # Use the public get() API — _collection.count() is a private method
    # that broke in earlier ChromaDB versions.
    existing = store.get(limit=1)
    if existing and existing.get("ids"):
        logger.info("ingest_skipped", reason="collection_already_populated")
        print("Collection already has documents. Skipping ingest.")
        return

    docs_path = Path(settings.docs_dir)
    txt_files = list(docs_path.glob("*.txt"))
    if not txt_files:
        logger.error("no_txt_files_found", path=str(docs_path))
        print(f"No .txt files found in {docs_path}")
        sys.exit(1)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_chunks = []

    for txt_file in txt_files:
        loader = TextLoader(str(txt_file), encoding="utf-8")
        raw_docs = loader.load()
        for doc in raw_docs:
            doc.metadata["source"] = txt_file.name
            doc.metadata["page"] = 0
        chunks = splitter.split_documents(raw_docs)
        all_chunks.extend(chunks)
        logger.info("file_chunked", file=txt_file.name, chunks=len(chunks))

    store.add_documents(all_chunks)
    logger.info("ingest_complete", total_chunks=len(all_chunks), files=len(txt_files))
    print(f"Ingested {len(all_chunks)} chunks from {len(txt_files)} files.")


if __name__ == "__main__":
    ingest()
