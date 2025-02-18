"""Tests for vector store implementations."""

from __future__ import annotations

import tempfile

import pytest

from src.vectorstore.base import BaseVectorStore, SearchResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embedding(dims: int = 10, seed: float = 1.0) -> list[float]:
    """Create a simple deterministic embedding vector."""
    import math

    return [math.sin(seed * (i + 1)) for i in range(dims)]


# ---------------------------------------------------------------------------
# test_base_is_abstract
# ---------------------------------------------------------------------------


class TestBaseIsAbstract:
    def test_base_is_abstract(self):
        """BaseVectorStore cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseVectorStore()


# ---------------------------------------------------------------------------
# Chroma tests
# ---------------------------------------------------------------------------


class TestChromaAddAndSearch:
    def test_chroma_add_and_search(self):
        """Add documents to Chroma and retrieve them via similarity search."""
        from src.vectorstore.chroma_store import ChromaVectorStore

        store = ChromaVectorStore(collection_name="test_add_search")
        dims = 10
        chunks = ["Hello world", "Goodbye world"]
        embeddings = [_make_embedding(dims, 1.0), _make_embedding(dims, 2.0)]
        metadatas = [{"source": "a.txt"}, {"source": "b.txt"}]

        ids = store.add_documents(chunks, embeddings, metadatas)

        assert len(ids) == 2
        assert store.get_document_count() == 2

        results = store.similarity_search(_make_embedding(dims, 1.0), top_k=2)
        assert len(results) >= 1
        assert isinstance(results[0], SearchResult)
        assert results[0].content in chunks


class TestChromaDelete:
    def test_chroma_delete(self):
        """Delete documents from Chroma by source document ID."""
        from src.vectorstore.chroma_store import ChromaVectorStore

        store = ChromaVectorStore(collection_name="test_delete")
        dims = 10
        chunks = ["Doc A chunk 1", "Doc A chunk 2", "Doc B chunk 1"]
        embeddings = [_make_embedding(dims, i) for i in range(1, 4)]
        metadatas = [
            {"source": "doc_a"},
            {"source": "doc_a"},
            {"source": "doc_b"},
        ]

        store.add_documents(chunks, embeddings, metadatas)
        assert store.get_document_count() == 3

        deleted = store.delete_by_document_id("doc_a")
        assert deleted == 2
        assert store.get_document_count() == 1


class TestChromaMetadataNoneStripped:
    def test_chroma_metadata_none_stripped(self):
        """None values in metadata are stripped before insertion."""
        from src.vectorstore.chroma_store import ChromaVectorStore

        store = ChromaVectorStore(collection_name="test_none_meta")
        dims = 10
        chunks = ["test"]
        embeddings = [_make_embedding(dims)]
        metadatas = [{"source": "a.txt", "author": None, "page": 1}]

        # Should not raise — None values are stripped internally.
        ids = store.add_documents(chunks, embeddings, metadatas)
        assert len(ids) == 1
        assert store.get_document_count() == 1


# ---------------------------------------------------------------------------
# FAISS tests
# ---------------------------------------------------------------------------


faiss_available = pytest.importorskip("faiss", reason="faiss-cpu not installed")


class TestFaissAddAndSearch:
    def test_faiss_add_and_search(self):
        """Add documents to FAISS and retrieve them via similarity search."""
        from src.vectorstore.faiss_store import FaissVectorStore

        dims = 10
        store = FaissVectorStore(dimensions=dims)

        chunks = ["Alpha", "Beta", "Gamma"]
        embeddings = [_make_embedding(dims, i) for i in range(1, 4)]
        metadatas = [{"source": f"doc_{i}"} for i in range(3)]

        ids = store.add_documents(chunks, embeddings, metadatas)
        assert len(ids) == 3
        assert store.get_document_count() == 3

        results = store.similarity_search(_make_embedding(dims, 1.0), top_k=2)
        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        # The most similar to embedding(seed=1) should be "Alpha" (seed=1).
        assert results[0].content == "Alpha"


class TestFaissDelete:
    def test_faiss_delete(self):
        """Delete documents from FAISS by source metadata."""
        from src.vectorstore.faiss_store import FaissVectorStore

        dims = 10
        store = FaissVectorStore(dimensions=dims)

        chunks = ["A1", "A2", "B1"]
        embeddings = [_make_embedding(dims, i) for i in range(1, 4)]
        metadatas = [{"source": "a"}, {"source": "a"}, {"source": "b"}]

        store.add_documents(chunks, embeddings, metadatas)
        assert store.get_document_count() == 3

        deleted = store.delete_by_document_id("a")
        assert deleted == 2
        assert store.get_document_count() == 1


class TestFaissPersistence:
    def test_faiss_persistence(self):
        """FAISS store persists and reloads from disk."""
        from src.vectorstore.faiss_store import FaissVectorStore

        dims = 10

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FaissVectorStore(dimensions=dims, persist_directory=tmpdir)
            chunks = ["Persisted"]
            embeddings = [_make_embedding(dims, 42.0)]
            metadatas = [{"source": "persist.txt"}]
            store.add_documents(chunks, embeddings, metadatas)
            assert store.get_document_count() == 1

            # Create a new store pointing at the same directory — should reload.
            store2 = FaissVectorStore(dimensions=dims, persist_directory=tmpdir)
            assert store2.get_document_count() == 1

            results = store2.similarity_search(_make_embedding(dims, 42.0), top_k=1)
            assert len(results) == 1
            assert results[0].content == "Persisted"


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class _FakeConfig:
    """Minimal config object for factory tests."""

    def __init__(self, provider: str, **kwargs):
        self.provider = provider
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestFactoryChroma:
    def test_factory_chroma(self):
        """get_vectorstore returns a ChromaVectorStore for 'chroma'."""
        from src.vectorstore.chroma_store import ChromaVectorStore
        from src.vectorstore.factory import get_vectorstore

        cfg = _FakeConfig("chroma", collection_name="factory_test")
        store = get_vectorstore(cfg)
        assert isinstance(store, ChromaVectorStore)


class TestFactoryInvalid:
    def test_factory_invalid_raises(self):
        """get_vectorstore raises ValueError for unknown providers."""
        from src.vectorstore.factory import get_vectorstore

        with pytest.raises(ValueError, match="Unknown vectorstore provider"):
            get_vectorstore(_FakeConfig("not_a_store"))
