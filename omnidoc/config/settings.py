"""
Settings — centralised configuration loaded from environment variables
and an optional ``config.yaml`` file.

Priority (highest → lowest):
  1. Environment variables  (e.g. OMNIDOC_QDRANT_HOST)
  2. config.yaml values
  3. Hard-coded defaults below
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime-configurable parameters for the OmniDoc system."""

    model_config = SettingsConfigDict(
        env_prefix="OMNIDOC_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Qdrant ──────────────────────────────────────────────────────────
    qdrant_host: str = Field(default="localhost", description="Qdrant server hostname.")
    qdrant_port: int = Field(default=6333, description="Qdrant REST port.")
    qdrant_collection: str = Field(default="omnidoc_vectors")
    qdrant_vector_size: int = Field(default=1024, description="Embedding dimension.")

    # ── Embedding ───────────────────────────────────────────────────────
    embedding_backend: Literal["bge", "openai"] = Field(
        default="bge",
        description="'bge' (local BAAI model) or 'openai' (API).",
    )
    embedding_model: str | None = Field(
        default=None,
        description="Override the default model name for the backend.",
    )
    embedding_batch_size: int = Field(default=32)

    # ── Chunking ────────────────────────────────────────────────────────
    chunk_sizes: list[int] = Field(
        default=[1024, 512, 128],
        description="Hierarchical chunk token sizes [parent, section, child].",
    )

    # ── Retrieval ───────────────────────────────────────────────────────
    retrieval_top_k: int = Field(
        default=100,
        description="Number of candidates from each retriever before fusion.",
    )
    rrf_k: int = Field(default=60, description="RRF smoothing constant.")
    rrf_dense_weight: float = Field(default=1.0)
    rrf_sparse_weight: float = Field(default=1.0)
    reranker_backend: Literal["cohere", "bge"] = Field(default="cohere")
    reranker_model: str | None = Field(default=None)
    reranker_top_n: int = Field(default=5, description="Final candidates sent to LLM.")

    # ── Generation ──────────────────────────────────────────────────────
    llm_backend: Literal["claude", "gpt4o"] = Field(default="claude")
    llm_model: str | None = Field(default=None)
    llm_temperature: float = Field(default=0.0)
    llm_max_tokens: int = Field(default=2048)

    # ── Storage paths ───────────────────────────────────────────────────
    bm25_index_path: Path = Field(default=Path("data/processed/bm25_index.pkl"))
    bm25_corpus_path: Path = Field(default=Path("data/processed/bm25_corpus.json"))
    metadata_db_path: Path = Field(default=Path("data/processed/metadata.db"))

    # ── API ─────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    # ── Evaluation ──────────────────────────────────────────────────────
    eval_output_dir: Path = Field(default=Path("data/evaluation"))
    eval_threshold_context_relevance: float = Field(default=0.85)
    eval_threshold_groundedness: float = Field(default=0.95)
    eval_threshold_answer_relevance: float = Field(default=0.90)

    # ── API Keys (loaded from env / .env) ───────────────────────────────
    openai_api_key: str | None = Field(default=None)
    anthropic_api_key: str | None = Field(default=None)
    cohere_api_key: str | None = Field(default=None)
