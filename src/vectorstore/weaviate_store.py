"""Weaviate-backed vector store with schema auto-creation."""

from __future__ import annotations

import uuid
from typing import Any

import structlog

from .base import BaseVectorStore, SearchResult

logger = structlog.get_logger(__name__)

_BATCH_SIZE = 100


class WeaviateVectorStore(BaseVectorStore):
    """Weaviate vector store.

    Requires: ``pip install weaviate-client``
    """

    def __init__(
        self,
        url: str = "http://localhost:8080",
        class_name: str = "Documents",
        api_key: str | None = None,
    ) -> None:
        import weaviate  # type: ignore[import-untyped]

        auth = weaviate.auth.AuthApiKey(api_key=api_key) if api_key else None
        self._client = weaviate.Client(url=url, auth_client_secret=auth)
        self._class_name = class_name

        self._ensure_schema()

        logger.info(
            "weaviate_store.init",
            url=url,
            class_name=class_name,
        )

    def _ensure_schema(self) -> None:
        """Create the Weaviate class if it does not exist."""
        if self._client.schema.exists(self._class_name):
            return

        class_obj = {
            "class": self._class_name,
            "vectorizer": "none",  # we supply our own vectors
            "properties": [
                {"name": "content", "dataType": ["text"]},
                {"name": "chunk_id", "dataType": ["string"]},
                {"name": "document_id", "dataType": ["string"]},
                {"name": "meta_json", "dataType": ["text"]},  # serialised metadata
            ],
        }
        self._client.schema.create_class(class_obj)
        logger.info("weaviate_store.schema_created", class_name=self._class_name)

    # -- mutations -----------------------------------------------------------

    def add_documents(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        import json

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in chunks]
        if metadatas is None:
            metadatas = [{} for _ in chunks]

        with self._client.batch as batch:
            batch.batch_size = _BATCH_SIZE
            for cid, chunk, emb, meta in zip(ids, chunks, embeddings, metadatas, strict=False):
                props = {
                    "content": chunk,
                    "chunk_id": cid,
                    "document_id": meta.get("document_id", ""),
                    "meta_json": json.dumps(meta),
                }
                batch.add_data_object(
                    data_object=props,
                    class_name=self._class_name,
                    vector=emb,
                    uuid=cid,
                )

        logger.info("weaviate_store.add_documents", count=len(chunks))
        return ids

    def delete_by_document_id(self, doc_id: str) -> int:
        """Delete all objects whose document_id matches *doc_id*."""
        where_filter = {
            "path": ["document_id"],
            "operator": "Equal",
            "valueString": doc_id,
        }
        result = self._client.batch.delete_objects(
            class_name=self._class_name,
            where=where_filter,
            output="verbose",
        )
        deleted = result.get("results", {}).get("successful", 0)
        logger.info(
            "weaviate_store.delete_by_document_id",
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
        import json

        query = (
            self._client.query.get(self._class_name, ["content", "chunk_id", "meta_json"])
            .with_near_vector({"vector": query_embedding})
            .with_limit(top_k)
            .with_additional(["distance", "id"])
        )

        if filters:
            operands = []
            for key, value in filters.items():
                operands.append({"path": [key], "operator": "Equal", "valueString": str(value)})
            if len(operands) == 1:
                query = query.with_where(operands[0])
            else:
                query = query.with_where({"operator": "And", "operands": operands})

        response = query.do()
        objects = response.get("data", {}).get("Get", {}).get(self._class_name, [])

        results: list[SearchResult] = []
        for obj in objects:
            meta = {}
            if obj.get("meta_json"):
                try:
                    meta = json.loads(obj["meta_json"])
                except (json.JSONDecodeError, TypeError):
                    pass

            distance = float(obj.get("_additional", {}).get("distance", 1.0))
            score = 1.0 - distance  # cosine distance to similarity

            results.append(
                SearchResult(
                    content=obj.get("content", ""),
                    metadata=meta,
                    score=max(score, 0.0),
                    chunk_id=obj.get("chunk_id", obj.get("_additional", {}).get("id", "")),
                )
            )
        return results

    # -- inspection ----------------------------------------------------------

    def list_documents(self) -> list[dict[str, Any]]:
        """Return unique documents by aggregating document_id values."""
        import json

        result = (
            self._client.query.get(self._class_name, ["document_id", "meta_json"])
            .with_limit(10_000)
            .do()
        )
        objects = result.get("data", {}).get("Get", {}).get(self._class_name, [])

        seen: dict[str, dict[str, Any]] = {}
        for obj in objects:
            doc_id = obj.get("document_id", "unknown")
            if doc_id not in seen:
                meta: dict[str, Any] = {}
                if obj.get("meta_json"):
                    try:
                        meta = json.loads(obj["meta_json"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                seen[doc_id] = meta
        return list(seen.values())

    def get_document_count(self) -> int:
        result = self._client.query.aggregate(self._class_name).with_meta_count().do()
        agg = result.get("data", {}).get("Aggregate", {}).get(self._class_name, [{}])
        return agg[0].get("meta", {}).get("count", 0) if agg else 0
