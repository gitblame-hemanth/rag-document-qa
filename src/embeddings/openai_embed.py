"""OpenAI embedding provider."""

from __future__ import annotations

import os
from typing import Any

import structlog

from embeddings.base import BaseEmbeddingProvider

logger = structlog.get_logger(__name__)

_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

_MAX_BATCH_SIZE = 2048


class OpenAIEmbeddings(BaseEmbeddingProvider):
    """Embedding provider using the OpenAI Embeddings API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        from openai import OpenAI

        self._model = model
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key must be provided via api_key param or OPENAI_API_KEY env var"
            )

        self._client = OpenAI(api_key=resolved_key)

        # Validate model
        if model not in _MODEL_DIMENSIONS:
            raise ValueError(f"Unsupported model '{model}'. Supported: {list(_MODEL_DIMENSIONS)}")

        # Custom dimensions only supported on v3 models
        if dimensions is not None and model == "text-embedding-ada-002":
            raise ValueError("text-embedding-ada-002 does not support custom dimensions")

        self._dimensions = dimensions or _MODEL_DIMENSIONS[model]
        self._custom_dimensions = dimensions

        logger.info(
            "openai_embeddings_initialized",
            model=self._model,
            dimensions=self._dimensions,
        )

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        result = self._call_api([text])
        return result[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents with automatic batch chunking."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), _MAX_BATCH_SIZE):
            batch = texts[i : i + _MAX_BATCH_SIZE]
            logger.debug(
                "openai_embed_batch",
                batch_index=i // _MAX_BATCH_SIZE,
                batch_size=len(batch),
                total=len(texts),
            )
            all_embeddings.extend(self._call_api(batch))
        return all_embeddings

    def get_dimensions(self) -> int:
        return self._dimensions

    def get_model_info(self) -> dict[str, Any]:
        return {
            "provider": "openai",
            "model": self._model,
            "dimensions": self._dimensions,
            "max_batch_size": _MAX_BATCH_SIZE,
            "supports_custom_dimensions": self._model != "text-embedding-ada-002",
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {
            "input": texts,
            "model": self._model,
        }
        if self._custom_dimensions is not None:
            kwargs["dimensions"] = self._custom_dimensions

        response = self._client.embeddings.create(**kwargs)

        # Sort by index to guarantee ordering matches input
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [d.embedding for d in sorted_data]
