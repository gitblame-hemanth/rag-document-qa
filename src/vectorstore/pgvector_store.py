"""PostgreSQL pgvector-backed vector store with connection pooling and JSONB filtering."""

from __future__ import annotations

import uuid
from typing import Any

import structlog

from .base import BaseVectorStore, SearchResult

logger = structlog.get_logger(__name__)


class PgVectorStore(BaseVectorStore):
    """pgvector-backed vector store.

    Requires: ``pip install pgvector psycopg2-binary sqlalchemy``
    """

    def __init__(
        self,
        connection_string: str,
        collection_name: str = "documents",
        embedding_dimensions: int = 1536,
        pool_size: int = 5,
        max_overflow: int = 10,
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
            pool_size=pool_size,
            max_overflow=max_overflow,
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
            pool_size=pool_size,
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

    def delete_by_document_id(self, doc_id: str) -> int:
        """Delete all chunks for a given document_id stored in JSONB metadata."""
        from sqlalchemy import text

        with self._engine.begin() as conn:
            result = conn.execute(
                text(f"DELETE FROM {self._table_name} WHERE metadata->>'document_id' = :doc_id"),
                {"doc_id": doc_id},
            )
            deleted = result.rowcount
            logger.info(
                "pgvector_store.delete_by_document_id",
                document_id=doc_id,
                deleted=deleted,
            )
            return deleted

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
