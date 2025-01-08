"""Pinecone-backed vector store with namespace support."""

from __future__ import annotations

import uuid
from typing import Any

import structlog

from .base import BaseVectorStore, SearchResult

logger = structlog.get_logger(__name__)

_UPSERT_BATCH_SIZE = 100


class PineconeVectorStore(BaseVectorStore):
    """Pinecone vector store.

    Requires: ``pip install pinecone-client``
    """

    def __init__(
        self,
        index_name: str,
        api_key: str | None = None,
        environment: str | None = None,
        namespace: str = "default",
    ) -> None:
        import os

        from pinecone import Pinecone  # type: ignore[import-untyped]

        api_key = api_key or os.environ.get("PINECONE_API_KEY", "")
        if not api_key:
            raise ValueError("Pinecone API key is required (param or PINECONE_API_KEY env var)")

        self._namespace = namespace
        self._pc = Pinecone(api_key=api_key)
        self._index = self._pc.Index(index_name)

        logger.info(
            "pinecone_store.init",
            index=index_name,
            namespace=namespace,
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

        vectors: list[dict[str, Any]] = []
        for cid, emb, chunk, meta in zip(ids, embeddings, chunks, metadatas, strict=False):
            record_meta = {k: v for k, v in meta.items() if v is not None}
            record_meta["_content"] = chunk
            vectors.append({"id": cid, "values": emb, "metadata": record_meta})

        # Batch upserts
        for start in range(0, len(vectors), _UPSERT_BATCH_SIZE):
            batch = vectors[start : start + _UPSERT_BATCH_SIZE]
            self._index.upsert(vectors=batch, namespace=self._namespace)

        logger.info("pinecone_store.add_documents", count=len(chunks))
        return ids

    def delete_by_document_id(self, doc_id: str) -> int:
        """Delete all vectors whose metadata.document_id matches *doc_id*.

        Pinecone does not return a deleted count directly so we query first,
        delete, and return the number of matched IDs.
        """
        results = self._index.query(
            vector=[0.0] * 1,  # dummy — overridden by filter-only query
            filter={"document_id": {"$eq": doc_id}},
            top_k=10_000,
            namespace=self._namespace,
            include_metadata=False,
        )
        matched_ids = [m["id"] for m in results.get("matches", [])]
        if matched_ids:
            self._index.delete(ids=matched_ids, namespace=self._namespace)
            logger.info(
                "pinecone_store.delete_by_document_id",
                document_id=doc_id,
                deleted=len(matched_ids),
            )
        return len(matched_ids)

    # -- queries -------------------------------------------------------------

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        kwargs: dict[str, Any] = {
            "vector": query_embedding,
            "top_k": top_k,
            "namespace": self._namespace,
            "include_metadata": True,
        }
        if filters:
            # Convert flat dict to Pinecone filter format
            pine_filter: dict[str, Any] = {}
            for key, value in filters.items():
                pine_filter[key] = {"$eq": value}
            kwargs["filter"] = pine_filter

        response = self._index.query(**kwargs)

        results: list[SearchResult] = []
        for match in response.get("matches", []):
            meta = dict(match.get("metadata", {}))
            content = meta.pop("_content", "")
            results.append(
                SearchResult(
                    content=content,
                    metadata=meta,
                    score=float(match.get("score", 0.0)),
                    chunk_id=match["id"],
                )
            )
        return results

    # -- inspection ----------------------------------------------------------

    def list_documents(self) -> list[dict[str, Any]]:
        """List unique documents by querying index stats.

        Pinecone does not support full scans natively, so this returns
        namespace-level stats rather than per-document metadata.
        """
        stats = self._index.describe_index_stats()
        ns_stats = stats.get("namespaces", {}).get(self._namespace, {})
        return [{"namespace": self._namespace, "vector_count": ns_stats.get("vector_count", 0)}]

    def get_document_count(self) -> int:
        stats = self._index.describe_index_stats()
        ns_stats = stats.get("namespaces", {}).get(self._namespace, {})
        return ns_stats.get("vector_count", 0)
