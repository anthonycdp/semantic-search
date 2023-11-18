"""Configuration settings for the semantic search system."""

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # Model settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384  # all-MiniLM-L6-v2 dimension

    # Search settings
    default_top_k: int = 10
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    hybrid_alpha: float = 0.5  # Weight for BM25 in hybrid (1-alpha for embeddings)

    # Performance settings
    batch_size: int = 32
    max_corpus_size: int = 100000

    # Data settings
    data_path: str = "data/documents.json"


settings = Settings()
