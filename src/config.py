# Configuration management
"""Configuration loader for RAG Document QA.

Loads from config/config.yaml with env var overrides.
Env var pattern: RAG_<SECTION>__<KEY> (e.g., RAG_LLM__MODEL=gpt-4o-mini)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
ENV_PREFIX = "RAG_"


class EmbeddingConfig(BaseModel):
    provider: Literal["openai", "huggingface"] = "openai"
    model: str = "text-embedding-3-small"
    dimensions: int = 1536


class LLMConfig(BaseModel):
    provider: Literal["openai", "bedrock"] = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 2048


class ChunkingConfig(BaseModel):
    strategy: Literal["fixed", "recursive", "semantic"] = "recursive"
    chunk_size: int = 1000
    overlap: int = 200


class VectorStoreConfig(BaseModel):
    provider: Literal["chroma", "pgvector"] = "chroma"
    collection_name: str = "documents"
    persist_directory: str = "./chroma_data"
    pgvector_connection_string: str = "postgresql+psycopg2://user:password@localhost:5432/rag_db"


class RetrievalConfig(BaseModel):
    top_k: int = 5
    search_type: Literal["similarity", "mmr", "hybrid"] = "mmr"
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8000"]
    )


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["json", "console"] = "json"


class AppConfig(BaseModel):
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _apply_env_overrides(data: dict) -> dict:
    """Override config values from environment variables.

    Pattern: RAG_<SECTION>__<KEY>=<value>
    Examples:
        RAG_LLM__MODEL=gpt-4o-mini
        RAG_EMBEDDING__PROVIDER=huggingface
        RAG_RETRIEVAL__TOP_K=10
        RAG_VECTORSTORE__PROVIDER=pgvector
    """
    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue

        parts = key[len(ENV_PREFIX) :].lower().split("__")
        if len(parts) != 2:
            continue

        section, field = parts
        if section not in data:
            data[section] = {}

        # Coerce types based on existing values
        existing = data[section].get(field)
        if isinstance(existing, bool):
            data[section][field] = value.lower() in ("true", "1", "yes")
        elif isinstance(existing, int):
            try:
                data[section][field] = int(value)
            except ValueError:
                data[section][field] = value
        elif isinstance(existing, float):
            try:
                data[section][field] = float(value)
            except ValueError:
                data[section][field] = value
        elif isinstance(existing, list):
            data[section][field] = [v.strip() for v in value.split(",")]
        else:
            data[section][field] = value

    return data


def load_config(config_path: Path | str | None = None) -> AppConfig:
    """Load configuration from YAML file with env var overrides."""
    path = Path(config_path) if config_path else CONFIG_PATH

    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}

    raw = _apply_env_overrides(raw)
    return AppConfig(**raw)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Get cached application configuration singleton."""
    return load_config()
