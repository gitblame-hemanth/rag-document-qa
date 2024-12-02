"""Abstract base class for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    """Abstract interface that all embedding providers must implement."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text into a vector.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents into vectors.

        Args:
            texts: A list of document texts to embed.

        Returns:
            A list of embedding vectors, one per input text.
        """

    @abstractmethod
    def get_dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors."""

    @abstractmethod
    def get_model_info(self) -> dict:
        """Return metadata about the embedding model.

        Returns:
            A dict with keys such as ``provider``, ``model``, ``dimensions``,
            and any provider-specific information.
        """
