"""Abstract base class for vector store backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    """Single search result returned by a vector store."""

    content: str
    metadata: dict[str, Any]
    score: float
    chunk_id: str


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
    def delete_by_document_id(self, doc_id: str) -> int:
        """Delete all chunks belonging to a given source document. Returns count deleted."""
        ...

    @abstractmethod
    def list_documents(self) -> list[dict[str, Any]]:
        """Return unique document-level metadata entries."""
        ...

    @abstractmethod
    def get_document_count(self) -> int:
        """Return the total number of stored chunks."""
        ...
