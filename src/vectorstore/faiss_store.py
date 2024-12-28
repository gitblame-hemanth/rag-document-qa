"""FAISS vector store implementation."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from .base import BaseVectorStore, SearchResult

logger = structlog.get_logger(__name__)


class FaissVectorStore(BaseVectorStore):
    """Vector store backed by FAISS with a JSON metadata sidecar."""

    def __init__(
        self,
        dimensions: int = 1536,
        persist_directory: str | None = None,
    ) -> None:
        import faiss

        self.dimensions = dimensions
        self.persist_directory = persist_directory

        self._index = faiss.IndexFlatIP(dimensions)  # inner-product (cosine on normalized vecs)
        self._documents: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._ids: list[str] = []

        if persist_directory and os.path.isfile(os.path.join(persist_directory, "index.faiss")):
            self._load(persist_directory)

        logger.info("faiss_store_initialized", dimensions=dimensions)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        metas = metadatas or [{} for _ in chunks]

        vectors = np.array(embeddings, dtype=np.float32)
        # Normalise for cosine similarity via inner product.
        faiss_module = self._get_faiss()
        faiss_module.normalize_L2(vectors)

        self._index.add(vectors)
        self._documents.extend(chunks)
        self._metadatas.extend(metas)
        self._ids.extend(doc_ids)

        if self.persist_directory:
            self._save(self.persist_directory)

        logger.info("faiss_documents_added", count=len(chunks))
        return doc_ids

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if self._index.ntotal == 0:
            return []

        query = np.array([query_embedding], dtype=np.float32)
        faiss_module = self._get_faiss()
        faiss_module.normalize_L2(query)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query, k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(
                SearchResult(
                    content=self._documents[idx],
                    metadata=self._metadatas[idx],
                    score=float(score),
                    chunk_id=self._ids[idx],
                )
            )
        return results

    def delete_by_document_id(self, doc_id: str) -> int:
        """Rebuild the index excluding chunks with matching source metadata."""
        indices_to_keep = [i for i, m in enumerate(self._metadatas) if m.get("source") != doc_id]
        removed = len(self._metadatas) - len(indices_to_keep)

        if removed == 0:
            return 0

        self._rebuild(indices_to_keep)
        if self.persist_directory:
            self._save(self.persist_directory)
        logger.info("faiss_documents_deleted", doc_id=doc_id, count=removed)
        return removed

    def list_documents(self) -> list[dict[str, Any]]:
        seen: dict[str, dict[str, Any]] = {}
        for meta in self._metadatas:
            if "source" in meta:
                source = meta["source"]
                if source not in seen:
                    seen[source] = dict(meta)
        return list(seen.values())

    def get_document_count(self) -> int:
        return self._index.ntotal

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save(self, directory: str) -> None:
        import faiss

        Path(directory).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "metadata.json"), "w") as f:
            json.dump(
                {"documents": self._documents, "metadatas": self._metadatas, "ids": self._ids},
                f,
            )

    def _load(self, directory: str) -> None:
        import faiss

        self._index = faiss.read_index(os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "metadata.json")) as f:
            data = json.load(f)
        self._documents = data["documents"]
        self._metadatas = data["metadatas"]
        self._ids = data["ids"]

    def _rebuild(self, keep_indices: list[int]) -> None:
        """Rebuild the FAISS index keeping only the given row indices."""
        import faiss

        if not keep_indices:
            self._index = faiss.IndexFlatIP(self.dimensions)
            self._documents = []
            self._metadatas = []
            self._ids = []
            return

        vectors = np.array(
            [self._index.reconstruct(int(i)) for i in keep_indices], dtype=np.float32
        )
        self._documents = [self._documents[i] for i in keep_indices]
        self._metadatas = [self._metadatas[i] for i in keep_indices]
        self._ids = [self._ids[i] for i in keep_indices]

        self._index = faiss.IndexFlatIP(self.dimensions)
        self._index.add(vectors)

    @staticmethod
    def _get_faiss():
        import faiss

        return faiss
