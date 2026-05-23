from langchain_chroma import Chroma
from app.services.embeddings import get_embeddings
from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

_vector_store: Chroma | None = None


def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        _vector_store = Chroma(
            collection_name=settings.chroma_collection,
            embedding_function=get_embeddings(),
            persist_directory=settings.chroma_persist_dir,
        )
        logger.info("chromadb_initialized", collection=settings.chroma_collection)
    return _vector_store
