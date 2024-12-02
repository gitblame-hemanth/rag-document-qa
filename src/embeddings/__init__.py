"""Embedding providers for document vectorization."""

from embeddings.base import BaseEmbeddingProvider
from embeddings.factory import get_embedding_provider

__all__ = [
    "BaseEmbeddingProvider",
    "get_embedding_provider",
]
