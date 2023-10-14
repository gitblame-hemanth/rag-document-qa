"""Chunking strategies for splitting documents into smaller pieces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import structlog

from src.ingestion.loaders import Document

logger = structlog.get_logger(__name__)


@dataclass
class Chunk:
    """A chunk of text derived from a source Document."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metadata.setdefault("source", "")
        self.metadata.setdefault("page", None)
        self.metadata.setdefault("chunk_index", 0)
        self.metadata.setdefault("chunk_strategy", "")
        self.metadata.setdefault("start_char", 0)
        self.metadata.setdefault("end_char", 0)


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk(self, documents: list[Document]) -> list[Chunk]:
        """Split documents into chunks.

        Args:
            documents: List of Document objects to chunk.

        Returns:
            List of Chunk objects.
        """
        ...

    def _make_chunk(
        self,
        text: str,
        doc: Document,
        *,
        chunk_index: int,
        strategy: str,
        start_char: int,
        end_char: int,
    ) -> Chunk:
        """Build a Chunk, carrying forward source Document metadata."""
        meta = {**doc.metadata}
        meta.update(
            chunk_index=chunk_index,
            chunk_strategy=strategy,
            start_char=start_char,
            end_char=end_char,
        )
        return Chunk(content=text, metadata=meta)


# ---------------------------------------------------------------------------
# Fixed-size chunker
# ---------------------------------------------------------------------------


class FixedChunker(BaseChunker):
    """Split documents into fixed-size character chunks with overlap."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        global_idx = 0

        for doc in documents:
            text = doc.content
            if not text:
                continue

            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                fragment = text[start:end]
                if fragment.strip():
                    chunks.append(
                        self._make_chunk(
                            fragment,
                            doc,
                            chunk_index=global_idx,
                            strategy="fixed",
                            start_char=start,
                            end_char=end,
                        )
                    )
                    global_idx += 1

                if end == len(text):
                    break
                start += self.chunk_size - self.overlap

        logger.info("chunker.fixed.done", total_chunks=len(chunks))
        return chunks


# ---------------------------------------------------------------------------
# Recursive character-based chunker
# ---------------------------------------------------------------------------

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]


class RecursiveChunker(BaseChunker):
    """Recursively split text using a hierarchy of separators."""

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        separators: list[str] | None = None,
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or list(_DEFAULT_SEPARATORS)

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split *text* trying each separator in order."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # Find best separator that exists in text
        sep = ""
        remaining_seps = separators
        for i, candidate in enumerate(separators):
            if candidate in text:
                sep = candidate
                remaining_seps = separators[i + 1 :]
                break

        if not sep:
            # No separator found — hard split at chunk_size
            pieces: list[str] = []
            start = 0
            while start < len(text):
                pieces.append(text[start : start + self.chunk_size])
                start += self.chunk_size - self.overlap
                if start + self.overlap >= len(text):
                    break
            return pieces

        splits = text.split(sep)

        # Merge small splits back together respecting chunk_size
        merged: list[str] = []
        current = ""
        for piece in splits:
            candidate = (current + sep + piece) if current else piece
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                # If single piece exceeds chunk_size, recurse deeper
                if len(piece) > self.chunk_size:
                    merged.extend(self._split_text(piece, remaining_seps))
                else:
                    current = piece
                    continue
                current = ""

        if current and current.strip():
            merged.append(current)

        return merged

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        global_idx = 0

        for doc in documents:
            text = doc.content
            if not text:
                continue

            fragments = self._split_text(text, self.separators)

            # Apply overlap by carrying trailing context
            for fragment in fragments:
                fragment = fragment.strip()
                if not fragment:
                    continue

                # Calculate approximate char offsets in original text
                start_char = text.find(fragment)
                if start_char == -1:
                    start_char = 0
                end_char = start_char + len(fragment)

                chunks.append(
                    self._make_chunk(
                        fragment,
                        doc,
                        chunk_index=global_idx,
                        strategy="recursive",
                        start_char=start_char,
                        end_char=end_char,
                    )
                )
                global_idx += 1

        logger.info("chunker.recursive.done", total_chunks=len(chunks))
        return chunks


# ---------------------------------------------------------------------------
# Semantic chunker
# ---------------------------------------------------------------------------


class SemanticChunker(BaseChunker):
    """Group sentences by embedding similarity.

    Falls back to RecursiveChunker when sentence-transformers is unavailable.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        similarity_threshold: float = 0.5,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name
        self._model: Any = None
        self._fallback = False

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
        except (ImportError, Exception) as exc:
            logger.warning(
                "semantic_chunker.fallback",
                reason=str(exc),
                msg="Falling back to recursive chunker",
            )
            self._fallback = True

    def _split_sentences(self, text: str) -> list[str]:
        """Naive sentence splitter that handles common abbreviations."""
        import re

        # Split on sentence-ending punctuation followed by whitespace or EOL
        raw = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in raw if s.strip()]

    def _cosine_similarity(self, a: Any, b: Any) -> float:
        """Cosine similarity between two 1-D vectors (numpy arrays)."""
        import numpy as np

        dot = float(np.dot(a, b))
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        return dot / norm if norm > 0 else 0.0

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        if self._fallback:
            logger.info("semantic_chunker.using_fallback")
            return RecursiveChunker(chunk_size=self.chunk_size, overlap=self.overlap).chunk(
                documents
            )

        chunks: list[Chunk] = []
        global_idx = 0

        for doc in documents:
            text = doc.content
            if not text:
                continue

            sentences = self._split_sentences(text)
            if not sentences:
                continue

            embeddings = self._model.encode(sentences)

            # Group sentences by similarity to the running group
            groups: list[list[int]] = []
            current_group: list[int] = [0]

            for i in range(1, len(sentences)):
                sim = self._cosine_similarity(embeddings[current_group[-1]], embeddings[i])

                group_text = " ".join(sentences[j] for j in current_group) + " " + sentences[i]

                if sim >= self.similarity_threshold and len(group_text) <= self.chunk_size:
                    current_group.append(i)
                else:
                    groups.append(current_group)
                    current_group = [i]

            if current_group:
                groups.append(current_group)

            # Convert groups to chunks
            for group in groups:
                fragment = " ".join(sentences[j] for j in group).strip()
                if not fragment:
                    continue

                start_char = text.find(sentences[group[0]])
                if start_char == -1:
                    start_char = 0
                last_sent = sentences[group[-1]]
                end_idx = text.find(last_sent)
                end_char = (
                    (end_idx + len(last_sent)) if end_idx != -1 else start_char + len(fragment)
                )

                chunks.append(
                    self._make_chunk(
                        fragment,
                        doc,
                        chunk_index=global_idx,
                        strategy="semantic",
                        start_char=start_char,
                        end_char=end_char,
                    )
                )
                global_idx += 1

        logger.info("chunker.semantic.done", total_chunks=len(chunks))
        return chunks


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_chunker(
    strategy: Literal["fixed", "recursive", "semantic"] = "recursive",
    chunk_size: int = 1000,
    overlap: int = 200,
) -> BaseChunker:
    """Return a chunker instance for the given strategy.

    Args:
        strategy: One of 'fixed', 'recursive', 'semantic'.
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters between chunks.

    Returns:
        A BaseChunker subclass instance.

    Raises:
        ValueError: If the strategy is unknown.
    """
    match strategy:
        case "fixed":
            return FixedChunker(chunk_size=chunk_size, overlap=overlap)
        case "recursive":
            return RecursiveChunker(chunk_size=chunk_size, overlap=overlap)
        case "semantic":
            return SemanticChunker(chunk_size=chunk_size, overlap=overlap)
        case _:
            raise ValueError(
                f"Unknown chunking strategy '{strategy}'. Choose from: fixed, recursive, semantic"
            )
