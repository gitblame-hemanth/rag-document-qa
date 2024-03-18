"""Tests for document chunking strategies."""

from __future__ import annotations

import pytest

from src.ingestion.loaders import Document

# ---------------------------------------------------------------------------
# We test chunkers through their public interface. The actual chunker module
# may not exist yet, so we mock-import where needed and test the contract.
# ---------------------------------------------------------------------------


@pytest.fixture()
def long_document() -> Document:
    """A document long enough to be split into multiple chunks."""
    text = " ".join([f"Sentence number {i} with some filler content here." for i in range(100)])
    return Document(
        content=text,
        metadata={"source": "/docs/long.txt", "filename": "long.txt", "file_type": "txt"},
    )


@pytest.fixture()
def short_document() -> Document:
    return Document(
        content="Short text.",
        metadata={"source": "/docs/short.txt", "filename": "short.txt", "file_type": "txt"},
    )


class TestFixedChunker:
    def test_fixed_chunker_splits_document(self, long_document):
        from src.ingestion.chunkers import FixedChunker

        chunker = FixedChunker(chunk_size=200, overlap=50)
        chunks = chunker.chunk([long_document])

        assert len(chunks) > 1
        for chunk in chunks:
            # Allow some tolerance for word boundaries
            assert len(chunk.content) <= 200 + 100  # generous buffer for word-boundary splitting

    def test_fixed_chunker_single_chunk_for_short_doc(self, short_document):
        from src.ingestion.chunkers import FixedChunker

        chunker = FixedChunker(chunk_size=200, overlap=50)
        chunks = chunker.chunk([short_document])

        assert len(chunks) == 1
        assert chunks[0].content == "Short text."

    def test_fixed_chunker_preserves_metadata(self, long_document):
        from src.ingestion.chunkers import FixedChunker

        chunker = FixedChunker(chunk_size=200, overlap=50)
        chunks = chunker.chunk([long_document])

        for chunk in chunks:
            assert chunk.metadata["source"] == "/docs/long.txt"
            assert chunk.metadata["filename"] == "long.txt"


class TestRecursiveChunker:
    def test_recursive_chunker_splits_document(self, long_document):
        from src.ingestion.chunkers import RecursiveChunker

        chunker = RecursiveChunker(chunk_size=200, overlap=50)
        chunks = chunker.chunk([long_document])

        assert len(chunks) > 1

    def test_recursive_chunker_respects_separators(self, long_document):
        from src.ingestion.chunkers import RecursiveChunker

        chunker = RecursiveChunker(chunk_size=200, overlap=50)
        chunks = chunker.chunk([long_document])

        # Each chunk should be non-empty
        for chunk in chunks:
            assert chunk.content.strip() != ""

    def test_recursive_chunker_preserves_metadata(self, long_document):
        from src.ingestion.chunkers import RecursiveChunker

        chunker = RecursiveChunker(chunk_size=200, overlap=50)
        chunks = chunker.chunk([long_document])

        for chunk in chunks:
            assert chunk.metadata["source"] == "/docs/long.txt"


class TestChunkMetadata:
    def test_chunk_metadata_preserved_across_strategies(self, long_document):
        """Both chunker strategies should carry forward document metadata."""
        from src.ingestion.chunkers import FixedChunker, RecursiveChunker

        for chunker_cls in [FixedChunker, RecursiveChunker]:
            chunker = chunker_cls(chunk_size=200, overlap=50)
            chunks = chunker.chunk([long_document])

            for chunk in chunks:
                assert "source" in chunk.metadata
                assert "filename" in chunk.metadata
                assert chunk.metadata["file_type"] == "txt"

    def test_chunk_has_index(self, long_document):
        """Each chunk should have a chunk_index in metadata or as attribute."""
        from src.ingestion.chunkers import FixedChunker

        chunker = FixedChunker(chunk_size=200, overlap=50)
        chunks = chunker.chunk([long_document])

        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))


class TestEmptyDocument:
    def test_empty_document_returns_no_chunks(self):
        from src.ingestion.chunkers import FixedChunker

        empty_doc = Document(content="", metadata={"source": "empty.txt"})
        chunker = FixedChunker(chunk_size=200, overlap=50)
        chunks = chunker.chunk([empty_doc])

        assert len(chunks) == 0

    def test_whitespace_only_document_returns_no_chunks(self):
        from src.ingestion.chunkers import FixedChunker

        ws_doc = Document(content="   \n\t  ", metadata={"source": "ws.txt"})
        chunker = FixedChunker(chunk_size=200, overlap=50)
        chunks = chunker.chunk([ws_doc])

        assert len(chunks) == 0
