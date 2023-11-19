"""Re-ranking — cross-encoder and pass-through implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

import structlog

from src.config import RetrievalConfig, get_config
from src.retrieval.vectorstore import SearchResult

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class BaseReranker(ABC):
    """Abstract reranker interface."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Re-score and re-sort *results* for *query*. Returns top_k."""
        ...


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------


class CrossEncoderReranker(BaseReranker):
    """Re-ranks using a sentence-transformers cross-encoder model.

    Each (query, document) pair is scored independently by the cross-encoder,
    then results are sorted by descending score.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

        self._model_name = model_name
        self._model = CrossEncoder(model_name)
        logger.info("cross_encoder_reranker.init", model=model_name)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        if not results:
            return []

        pairs = [[query, r.content] for r in results]
        scores = self._model.predict(pairs)

        scored = sorted(
            zip(results, scores, strict=False),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        top_k = top_k or len(results)
        reranked = [
            SearchResult(
                content=r.content,
                metadata=r.metadata,
                score=float(s),
                chunk_id=r.chunk_id,
            )
            for r, s in scored[:top_k]
        ]

        logger.info(
            "cross_encoder_reranker.rerank",
            input_count=len(results),
            output_count=len(reranked),
            model=self._model_name,
        )
        return reranked


# ---------------------------------------------------------------------------
# No-op reranker
# ---------------------------------------------------------------------------


class NoOpReranker(BaseReranker):
    """Pass-through reranker — returns input unchanged (optionally truncated)."""

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        if top_k is not None:
            return results[:top_k]
        return results


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_reranker(config: RetrievalConfig | None = None) -> BaseReranker:
    """Factory: build a reranker from config."""
    if config is None:
        config = get_config().retrieval

    if not config.reranker_enabled:
        logger.info("reranker.factory", type="noop")
        return NoOpReranker()

    logger.info("reranker.factory", type="cross_encoder", model=config.reranker_model)
    return CrossEncoderReranker(model_name=config.reranker_model)
