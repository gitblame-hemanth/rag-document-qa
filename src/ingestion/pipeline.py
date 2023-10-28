# Async document processing pipeline
"""Ingestion pipeline — load, chunk, embed, and store documents."""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import structlog

from src.config import get_config
from src.ingestion.chunkers import Chunk, get_chunker
from src.ingestion.loaders import get_loader

logger = structlog.get_logger(__name__)


@dataclass
class IngestResult:
    """Summary of a single file ingestion."""

    filename: str
    num_chunks: int
    chunk_strategy: str
    duration_seconds: float
    document_id: str
    error: str | None = None


class IngestPipeline:
    """Orchestrates document ingestion: load -> chunk -> embed -> store.

    Args:
        vectorstore: An object exposing ``add_documents(docs, embeddings, metadatas, ids)``
                      and ``get(where)`` async or sync methods.  Typically a Chroma/pgvector
                      wrapper from ``src.retrieval``.
        embedding_fn: Async callable that takes a list of texts and returns a list of
                      embedding vectors (list[list[float]]).  When *None*, a stub that
                      returns zero vectors is used (useful for testing without an API key).
    """

    def __init__(
        self,
        vectorstore: Any = None,
        embedding_fn: Any = None,
    ) -> None:
        self._cfg = get_config()
        self._vectorstore = vectorstore
        self._embedding_fn = embedding_fn
        self._ingested_hashes: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest_file(
        self,
        file_path: str | Path,
        chunking_strategy: Literal["fixed", "recursive", "semantic"] | None = None,
    ) -> IngestResult:
        """Ingest a single file end-to-end.

        Args:
            file_path: Path to the document.
            chunking_strategy: Override the default strategy from config.

        Returns:
            IngestResult with ingestion metadata.
        """
        path = Path(file_path)
        start = time.perf_counter()
        document_id = uuid4().hex

        strategy = chunking_strategy or self._cfg.chunking.strategy
        log = logger.bind(file=path.name, document_id=document_id, strategy=strategy)

        try:
            # --- Deduplication via content hash ---
            file_hash = self._hash_file(path)
            if file_hash in self._ingested_hashes:
                elapsed = time.perf_counter() - start
                log.info("ingest.skipped.duplicate", hash=file_hash)
                return IngestResult(
                    filename=path.name,
                    num_chunks=0,
                    chunk_strategy=strategy,
                    duration_seconds=round(elapsed, 3),
                    document_id=document_id,
                    error="duplicate — file already ingested",
                )

            # --- Load ---
            log.info("ingest.loading")
            loader = get_loader(path)
            documents = await asyncio.to_thread(loader.load, path)
            log.info("ingest.loaded", doc_fragments=len(documents))

            # --- Chunk ---
            log.info("ingest.chunking")
            chunker = get_chunker(
                strategy=strategy,
                chunk_size=self._cfg.chunking.chunk_size,
                overlap=self._cfg.chunking.overlap,
            )
            chunks = await asyncio.to_thread(chunker.chunk, documents)
            log.info("ingest.chunked", num_chunks=len(chunks))

            if not chunks:
                elapsed = time.perf_counter() - start
                return IngestResult(
                    filename=path.name,
                    num_chunks=0,
                    chunk_strategy=strategy,
                    duration_seconds=round(elapsed, 3),
                    document_id=document_id,
                    error="no chunks produced",
                )

            # --- Embed ---
            log.info("ingest.embedding", num_chunks=len(chunks))
            texts = [c.content for c in chunks]
            embeddings = await self._embed(texts)

            # --- Store ---
            log.info("ingest.storing")
            await self._store(chunks, embeddings, document_id)

            # Mark hash as ingested
            self._ingested_hashes.add(file_hash)

            elapsed = time.perf_counter() - start
            log.info("ingest.complete", duration=round(elapsed, 3), num_chunks=len(chunks))
            return IngestResult(
                filename=path.name,
                num_chunks=len(chunks),
                chunk_strategy=strategy,
                duration_seconds=round(elapsed, 3),
                document_id=document_id,
            )

        except Exception as exc:
            elapsed = time.perf_counter() - start
            log.error("ingest.failed", error=str(exc))
            return IngestResult(
                filename=path.name,
                num_chunks=0,
                chunk_strategy=strategy,
                duration_seconds=round(elapsed, 3),
                document_id=document_id,
                error=str(exc),
            )

    async def ingest_directory(
        self,
        dir_path: str | Path,
        chunking_strategy: Literal["fixed", "recursive", "semantic"] | None = None,
        extensions: set[str] | None = None,
    ) -> list[IngestResult]:
        """Ingest every supported file in a directory (non-recursive by default).

        Args:
            dir_path: Directory containing documents.
            chunking_strategy: Override per-file strategy.
            extensions: Limit to these extensions (e.g. ``{".pdf", ".docx"}``).
                        Defaults to all supported types.

        Returns:
            List of IngestResult, one per file processed.
        """
        dir_ = Path(dir_path)
        if not dir_.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_}")

        supported = extensions or {".pdf", ".docx", ".txt", ".md", ".markdown"}
        files = sorted(f for f in dir_.iterdir() if f.is_file() and f.suffix.lower() in supported)

        if not files:
            logger.warning("ingest_directory.no_files", directory=str(dir_))
            return []

        logger.info("ingest_directory.start", directory=str(dir_), file_count=len(files))

        results: list[IngestResult] = []
        for idx, fpath in enumerate(files, start=1):
            logger.info(
                "ingest_directory.progress",
                file=fpath.name,
                current=idx,
                total=len(files),
            )
            result = await self.ingest_file(fpath, chunking_strategy=chunking_strategy)
            results.append(result)

        succeeded = sum(1 for r in results if r.error is None)
        logger.info(
            "ingest_directory.complete",
            total=len(results),
            succeeded=succeeded,
            failed=len(results) - succeeded,
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_file(path: Path) -> str:
        """SHA-256 hex digest of file contents for deduplication."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 16), b""):
                h.update(block)
        return h.hexdigest()

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Uses the caller-supplied embedding function if available,
        otherwise returns zero vectors (for testing without an API key).
        """
        if self._embedding_fn is not None:
            if asyncio.iscoroutinefunction(self._embedding_fn):
                return await self._embedding_fn(texts)
            return await asyncio.to_thread(self._embedding_fn, texts)

        logger.warning("ingest.embed.stub", msg="No embedding_fn provided — returning zero vectors")
        dim = self._cfg.embedding.dimensions
        return [[0.0] * dim for _ in texts]

    async def _store(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        document_id: str,
    ) -> None:
        """Persist chunks + embeddings to the vectorstore."""
        if self._vectorstore is None:
            logger.warning("ingest.store.stub", msg="No vectorstore configured — skipping storage")
            return

        ids = [f"{document_id}_{c.metadata['chunk_index']}" for c in chunks]
        metadatas = [c.metadata for c in chunks]
        texts = [c.content for c in chunks]

        # Support both sync and async vectorstore interfaces
        add = getattr(self._vectorstore, "add_documents", None) or getattr(
            self._vectorstore, "add", None
        )
        if add is None:
            raise TypeError("vectorstore must expose an add_documents() or add() method")

        kwargs: dict[str, Any] = {
            "documents": texts,
            "embeddings": embeddings,
            "metadatas": metadatas,
            "ids": ids,
        }

        if asyncio.iscoroutinefunction(add):
            await add(**kwargs)
        else:
            await asyncio.to_thread(add, **kwargs)

        logger.info("ingest.stored", count=len(ids))
