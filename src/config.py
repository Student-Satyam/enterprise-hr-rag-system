"""Centralized configuration for the Enterprise RAG System."""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or a .env file."""

    # API Keys & Observability (loaded from .env or environment variables)
    groq_api_key: str = ""
    gemini_api_key: str = ""
    google_api_key: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Storage & Data Paths
    data_dir: str = "data"
    vector_store_dir: str = "vector_store"
    chroma_collection_name: str = "hr_policies"
    bm25_docs_filename: str = "bm25_docs.pkl"

    # Chunking Settings (1000/200 chunking; context blocks capped at 1200 chars)
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_context_chunk_chars: int = 1200

    # Lightweight ONNX Embedding & Reranker Models
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    reranker_model_name: str = "ms-marco-MiniLM-L-12-v2"

    # Retrieval & Cache Tuning
    retriever_top_k: int = 8
    reranker_top_n: int = 4
    cache_similarity_threshold: float = 0.90

    # LLM Models
    fast_llm_model: str = "openai/gpt-oss-20b"
    complex_llm_model: str = "openai/gpt-oss-120b"
    router_rewrite_model: str = "llama-3.1-8b-instant"

    # Backend API URL for Streamlit UI
    api_url: str = "http://localhost:8000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def vector_store_path(self) -> Path:
        """Return resolved Path object for persistent vector store directory."""
        return Path(self.vector_store_dir)

    @property
    def bm25_docs_path(self) -> Path:
        """Return resolved Path object for pickled BM25 documents."""
        return self.vector_store_path / self.bm25_docs_filename


@lru_cache
def get_settings() -> Settings:
    """Cache and return a single Settings instance across the application."""
    return Settings()