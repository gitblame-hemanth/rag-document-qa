"""Factory for creating vector store instances."""

from __future__ import annotations

from typing import Any

import structlog

from .base import BaseVectorStore

logger = structlog.get_logger(__name__)


def get_vectorstore(config: Any) -> BaseVectorStore:
    """Instantiate the correct vector store backend from configuration.

    Accepts either:
    - An object with a ``vectorstore`` attribute (e.g. ``AppConfig``)
    - A ``VectorStoreConfig``-like object with a ``provider`` field directly

    Provider-specific fields are read via ``getattr`` with sensible defaults.
    """
    vs_cfg = config.vectorstore if hasattr(config, "vectorstore") else config
    provider: str = getattr(vs_cfg, "provider", "chroma").lower()

    logger.info("vectorstore.factory", provider=provider)

    if provider == "chroma":
        from .chroma_store import ChromaVectorStore

        return ChromaVectorStore(
            collection_name=getattr(vs_cfg, "collection_name", "documents"),
            persist_directory=getattr(vs_cfg, "persist_directory", "./chroma_data"),
        )

    if provider == "pgvector":
        from .pgvector_store import PgVectorStore

        return PgVectorStore(
            connection_string=getattr(
                vs_cfg,
                "pgvector_connection_string",
                "postgresql+psycopg2://user:password@localhost:5432/rag_db",
            ),
            collection_name=getattr(vs_cfg, "collection_name", "documents"),
            embedding_dimensions=getattr(vs_cfg, "embedding_dimensions", 1536),
        )

    if provider == "pinecone":
        from .pinecone_store import PineconeVectorStore

        return PineconeVectorStore(
            index_name=getattr(vs_cfg, "index_name", "documents"),
            api_key=getattr(vs_cfg, "pinecone_api_key", None),
            environment=getattr(vs_cfg, "pinecone_environment", None),
            namespace=getattr(vs_cfg, "namespace", "default"),
        )

    if provider == "weaviate":
        from .weaviate_store import WeaviateVectorStore

        return WeaviateVectorStore(
            url=getattr(vs_cfg, "weaviate_url", "http://localhost:8080"),
            class_name=getattr(vs_cfg, "class_name", "Documents"),
            api_key=getattr(vs_cfg, "weaviate_api_key", None),
        )

    if provider == "faiss":
        from .faiss_store import FaissVectorStore

        return FaissVectorStore(
            dimensions=getattr(vs_cfg, "embedding_dimensions", 1536),
            persist_directory=getattr(vs_cfg, "faiss_index_path", None),
        )

    raise ValueError(
        f"Unknown vectorstore provider: '{provider}'. "
        "Supported: chroma, pgvector, pinecone, weaviate, faiss."
    )
