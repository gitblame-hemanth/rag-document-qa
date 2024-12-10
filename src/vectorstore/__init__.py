"""Vector store abstraction layer.

Provides a unified interface for multiple vector database backends:
Chroma, pgvector, Pinecone, Weaviate, and FAISS.
"""

from .base import BaseVectorStore, SearchResult
from .chroma_store import ChromaVectorStore
from .factory import get_vectorstore
from .faiss_store import FaissVectorStore
from .pgvector_store import PgVectorStore
from .pinecone_store import PineconeVectorStore
from .weaviate_store import WeaviateVectorStore

__all__ = [
    "BaseVectorStore",
    "ChromaVectorStore",
    "FaissVectorStore",
    "PgVectorStore",
    "PineconeVectorStore",
    "SearchResult",
    "WeaviateVectorStore",
    "get_vectorstore",
]
