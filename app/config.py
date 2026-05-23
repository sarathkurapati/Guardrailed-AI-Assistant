from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env")
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"
    embedding_model: str = "nomic-embed-text"

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "enterprise_docs"

    # Graph thresholds
    max_generation_attempts: int = 3
    toxicity_threshold: float = 0.5
    toxicity_hard_block_threshold: float = 0.8
    grounding_similarity_threshold: float = 0.75
    grounding_failure_ratio: float = 0.30
    hitl_confidence_threshold: float = 0.5
    injection_score_threshold: float = 0.7
    scope_similarity_threshold: float = 0.35

    # Paths
    hitl_queue_dir: str = "./data/hitl_queue"
    audit_log_dir: str = "./data/audit_logs"
    docs_dir: str = "./docs/sample_documents"
    chroma_dir: str = "./data/chroma"

    # LLM call timeout in seconds
    llm_timeout: int = 10


settings = Settings()
