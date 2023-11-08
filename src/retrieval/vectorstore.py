"""Vector store abstraction and embedding providers.

Supports Chroma (default) and PgVector backends with a unified interface.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

from src.config import EmbeddingConfig, VectorStoreConfig, get_config

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """Single search result returned by a vector store."""

    content: str
    metadata: dict[str, Any]
    score: float
    chunk_id: str


# ---------------------------------------------------------------------------
# Embedding provider
# ---------------------------------------------------------------------------


class EmbeddingProvider:
    """Unified wrapper around OpenAI / HuggingFace embedding models."""

    def __init__(self, model: Any) -> None:
        self._model = model

    # -- factories -----------------------------------------------------------

    @staticmethod
    def get_openai_embeddings(model: str = "text-embedding-3-small") -> EmbeddingProvider:
        """Create provider backed by langchain-openai embeddings."""
        from langchain_openai import OpenAIEmbeddings  # type: ignore[import-untyped]

        return EmbeddingProvider(OpenAIEmbeddings(model=model))

    @staticmethod
    def get_huggingface_embeddings(model: str = "all-MiniLM-L6-v2") -> EmbeddingProvider:
        """Create provider backed by sentence-transformers."""
        from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore[import-untyped]

        return EmbeddingProvider(HuggingFaceEmbeddings(model_name=model))

    # -- public API ----------------------------------------------------------

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._model.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document strings."""
        return self._model.embed_documents(texts)


def get_embeddings(config: EmbeddingConfig | None = None) -> EmbeddingProvider:
    """Factory: build an EmbeddingProvider from config."""
    if config is None:
        config = get_config().embedding

    if config.provider == "openai":
        logger.info("embedding_provider.init", provider="openai", model=config.model)
        return EmbeddingProvider.get_openai_embeddings(model=config.model)

    if config.provider == "huggingface":
        logger.info("embedding_provider.init", provider="huggingface", model=config.model)
        return EmbeddingProvider.get_huggingface_embeddings(model=config.model)

    raise ValueError(f"Unknown embedding provider: {config.provider}")


# ---------------------------------------------------------------------------
# Base vector store
# ---------------------------------------------------------------------------


class BaseVectorStore(ABC):
    """Abstract base for all vector store backends."""

    @abstractmethod
    def add_documents(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Insert document chunks with precomputed embeddings. Returns chunk IDs."""
        ...

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return top-k most similar chunks."""
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete chunks by their IDs."""
        ...

    @abstractmethod
    def list_documents(self) -> list[dict[str, Any]]:
        """Return unique document-level metadata entries."""
        ...

    @abstractmethod
    def get_document_count(self) -> int:
        """Return the total number of stored chunks."""
        ...


# ---------------------------------------------------------------------------
# Chroma implementation
# ---------------------------------------------------------------------------


class ChromaStore(BaseVectorStore):
    """ChromaDB-backed vector store with persistent storage."""

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./chroma_data",
    ) -> None:
        import chromadb  # type: ignore[import-untyped]

        self._persist_directory = persist_directory
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "chroma_store.init",
            collection=collection_name,
            persist_directory=persist_directory,
            existing_count=self._collection.count(),
        )

    # -- mutations -----------------------------------------------------------

    def add_documents(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in chunks]
        if metadatas is None:
            metadatas = [{} for _ in chunks]

        # ChromaDB rejects None values in metadata — strip them
        metadatas = [{k: v for k, v in m.items() if v is not None} for m in metadatas]

        # Chroma has a per-call batch limit; chunk to 5000
        batch_size = 5000
        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            self._collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=chunks[start:end],
                metadatas=metadatas[start:end],
            )

        logger.info("chroma_store.add_documents", count=len(chunks))
        return ids

    def delete(self, ids: list[str]) -> None:
        self._collection.delete(ids=ids)
        logger.info("chroma_store.delete", count=len(ids))

    def delete_by_document_id(self, doc_id: str) -> None:
        """Delete all chunks belonging to a given source document."""
        results = self._collection.get(where={"document_id": doc_id})
        if results["ids"]:
            self._collection.delete(ids=results["ids"])
            logger.info(
                "chroma_store.delete_by_document_id",
                document_id=doc_id,
                deleted=len(results["ids"]),
            )

    # -- queries -------------------------------------------------------------

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self._collection.count() or top_k),
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            kwargs["where"] = filters

        results = self._collection.query(**kwargs)

        out: list[SearchResult] = []
        for doc, meta, dist, cid in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
            strict=False,
        ):
            # Chroma cosine distance ∈ [0,2]; convert to similarity score ∈ [0,1]
            score = 1.0 - (dist / 2.0)
            out.append(SearchResult(content=doc, metadata=meta, score=score, chunk_id=cid))
        return out

    def mmr_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        lambda_mult: float = 0.5,
        fetch_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Maximal Marginal Relevance search.

        Fetches ``fetch_k`` candidates then iteratively selects ``top_k``
        results balancing relevance (lambda_mult) and diversity.
        """
        if fetch_k is None:
            fetch_k = max(top_k * 4, 20)

        candidates = self.similarity_search(query_embedding, top_k=fetch_k, filters=filters)
        if not candidates:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)

        # Re-fetch embeddings for candidates
        fetched = self._collection.get(
            ids=[c.chunk_id for c in candidates],
            include=["embeddings"],
        )
        candidate_embeddings = np.array(fetched["embeddings"], dtype=np.float32)

        # Normalise for cosine similarity
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-10
        candidate_normed = candidate_embeddings / norms

        selected_indices: list[int] = []
        remaining = list(range(len(candidates)))

        for _ in range(min(top_k, len(candidates))):
            best_idx = -1
            best_score = -float("inf")

            for idx in remaining:
                relevance = float(np.dot(candidate_normed[idx], query_norm))

                diversity = 0.0
                if selected_indices:
                    sims = candidate_normed[idx] @ candidate_normed[selected_indices].T
                    diversity = float(np.max(sims))

                mmr_score = lambda_mult * relevance - (1.0 - lambda_mult) * diversity
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx == -1:
                break
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        return [
            SearchResult(
                content=candidates[i].content,
                metadata=candidates[i].metadata,
                score=candidates[i].score,
                chunk_id=candidates[i].chunk_id,
            )
            for i in selected_indices
        ]

    # -- inspection ----------------------------------------------------------

    def list_documents(self) -> list[dict[str, Any]]:
        """Return deduplicated document-level metadata."""
        all_meta = self._collection.get(include=["metadatas"])
        seen: dict[str, dict[str, Any]] = {}
        for meta in all_meta["metadatas"] or []:
            doc_id = meta.get("document_id", meta.get("source", "unknown"))
            if doc_id not in seen:
                seen[doc_id] = meta
        return list(seen.values())

    def get_document_count(self) -> int:
        return self._collection.count()


# ---------------------------------------------------------------------------
# PgVector implementation
# ---------------------------------------------------------------------------


class PgVectorStore(BaseVectorStore):
    """pgvector-backed vector store (requires running PostgreSQL with pgvector).

    Requires: ``pip install pgvector psycopg2-binary sqlalchemy``
    """

    def __init__(
        self,
        connection_string: str,
        collection_name: str = "documents",
        embedding_dimensions: int = 1536,
    ) -> None:
        from pgvector.sqlalchemy import Vector  # type: ignore[import-untyped]
        from sqlalchemy import (
            Column,
            MetaData,
            String,
            Table,
            Text,
            create_engine,
            text,
        )
        from sqlalchemy.dialects.postgresql import JSONB
        from sqlalchemy.pool import QueuePool

        self._engine = create_engine(
            connection_string,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
        self._table_name = collection_name
        self._dimensions = embedding_dimensions

        meta = MetaData()
        self._table = Table(
            collection_name,
            meta,
            Column("id", String, primary_key=True),
            Column("content", Text, nullable=False),
            Column("metadata", JSONB, default={}),
            Column("embedding", Vector(embedding_dimensions), nullable=False),
        )

        with self._engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            meta.create_all(conn)

        logger.info(
            "pgvector_store.init",
            collection=collection_name,
            dimensions=embedding_dimensions,
        )

    # -- mutations -----------------------------------------------------------

    def add_documents(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in chunks]
        if metadatas is None:
            metadatas = [{} for _ in chunks]

        rows = [
            {"id": cid, "content": chunk, "metadata": meta, "embedding": emb}
            for cid, chunk, meta, emb in zip(ids, chunks, metadatas, embeddings, strict=False)
        ]

        with self._engine.begin() as conn:
            conn.execute(self._table.insert(), rows)

        logger.info("pgvector_store.add_documents", count=len(chunks))
        return ids

    def delete(self, ids: list[str]) -> None:
        with self._engine.begin() as conn:
            conn.execute(self._table.delete().where(self._table.c.id.in_(ids)))
        logger.info("pgvector_store.delete", count=len(ids))

    def delete_by_document_id(self, doc_id: str) -> None:
        """Delete all chunks for a given document_id stored in metadata."""
        from sqlalchemy import text

        with self._engine.begin() as conn:
            result = conn.execute(
                text(f"DELETE FROM {self._table_name} WHERE metadata->>'document_id' = :doc_id"),
                {"doc_id": doc_id},
            )
            logger.info(
                "pgvector_store.delete_by_document_id",
                document_id=doc_id,
                deleted=result.rowcount,
            )

    # -- queries -------------------------------------------------------------

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        from sqlalchemy import select, text

        emb_literal = str(query_embedding)
        distance_expr = text(f"embedding <=> '{emb_literal}'::vector")

        stmt = (
            select(
                self._table.c.id,
                self._table.c.content,
                self._table.c.metadata,
                distance_expr.label("distance"),
            )
            .order_by("distance")
            .limit(top_k)
        )

        if filters:
            for key, value in filters.items():
                stmt = stmt.where(text(f"metadata->>'{key}' = :fval_{key}")).params(
                    **{f"fval_{key}": str(value)}
                )

        with self._engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()

        results: list[SearchResult] = []
        for row in rows:
            score = 1.0 - float(row.distance)
            results.append(
                SearchResult(
                    content=row.content,
                    metadata=row.metadata or {},
                    score=max(score, 0.0),
                    chunk_id=row.id,
                )
            )
        return results

    # -- inspection ----------------------------------------------------------

    def list_documents(self) -> list[dict[str, Any]]:
        from sqlalchemy import text

        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT DISTINCT ON (metadata->>'document_id') metadata "
                    f"FROM {self._table_name} "
                    f"WHERE metadata->>'document_id' IS NOT NULL"
                )
            ).fetchall()
        return [row[0] for row in rows]

    def get_document_count(self) -> int:
        from sqlalchemy import func, select

        with self._engine.connect() as conn:
            count = conn.execute(select(func.count()).select_from(self._table)).scalar()
        return count or 0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_vectorstore(config: VectorStoreConfig | None = None) -> BaseVectorStore:
    """Factory: instantiate the correct vector store backend from config."""
    if config is None:
        config = get_config().vectorstore

    if config.provider == "chroma":
        logger.info("vectorstore.factory", provider="chroma")
        return ChromaStore(
            collection_name=config.collection_name,
            persist_directory=config.persist_directory,
        )

    if config.provider == "pgvector":
        logger.info("vectorstore.factory", provider="pgvector")
        return PgVectorStore(
            connection_string=config.pgvector_connection_string,
            collection_name=config.collection_name,
        )

    raise ValueError(f"Unknown vectorstore provider: {config.provider}")
