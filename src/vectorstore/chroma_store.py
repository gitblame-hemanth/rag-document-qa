"""ChromaDB vector store implementation with MMR support."""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import structlog

from .base import BaseVectorStore, SearchResult

logger = structlog.get_logger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """Vector store backed by ChromaDB."""

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str | None = None,
    ) -> None:
        import chromadb

        self.collection_name = collection_name

        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "chroma_store_initialized",
            collection=collection_name,
            persist=persist_directory,
        )

    def add_documents(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        if not chunks:
            return []

        doc_ids = ids or [str(uuid.uuid4()) for _ in chunks]

        # ChromaDB does not accept None values in metadata dicts.
        clean_metadatas: list[dict[str, Any]] | None = None
        if metadatas:
            clean_metadatas = [{k: v for k, v in m.items() if v is not None} for m in metadatas]

        # Chroma has a per-call batch limit; chunk to 5000.
        batch_size = 5000
        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            self._collection.add(
                documents=chunks[start:end],
                embeddings=embeddings[start:end],
                metadatas=clean_metadatas[start:end] if clean_metadatas else None,
                ids=doc_ids[start:end],
            )
        logger.info("chroma_documents_added", count=len(chunks))
        return doc_ids

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        if filters:
            kwargs["where"] = filters

        results = self._collection.query(**kwargs)

        search_results: list[SearchResult] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for doc, meta, dist, chunk_id in zip(documents, metadatas, distances, ids):
            # Chroma returns distances; convert to similarity score.
            score = 1.0 - dist
            search_results.append(
                SearchResult(
                    content=doc,
                    metadata=meta or {},
                    score=score,
                    chunk_id=chunk_id,
                )
            )
        return search_results

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

    def delete_by_document_id(self, doc_id: str) -> int:
        # Chroma filters on metadata to find chunks belonging to a source doc.
        results = self._collection.get(where={"source": doc_id})
        ids_to_delete = results.get("ids", [])
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
        logger.info("chroma_documents_deleted", doc_id=doc_id, count=len(ids_to_delete))
        return len(ids_to_delete)

    def list_documents(self) -> list[dict[str, Any]]:
        all_data = self._collection.get()
        seen: dict[str, dict[str, Any]] = {}
        for meta in all_data.get("metadatas", []):
            if meta and "source" in meta:
                source = meta["source"]
                if source not in seen:
                    seen[source] = dict(meta)
        return list(seen.values())

    def get_document_count(self) -> int:
        return self._collection.count()
