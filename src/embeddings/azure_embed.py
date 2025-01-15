"""Azure OpenAI embedding provider."""

from __future__ import annotations

import os
from typing import Any

import structlog

from embeddings.base import BaseEmbeddingProvider

logger = structlog.get_logger(__name__)

_MAX_BATCH_SIZE = 2048


class AzureOpenAIEmbeddings(BaseEmbeddingProvider):
    """Embedding provider using Azure OpenAI Service."""

    def __init__(
        self,
        deployment_name: str,
        endpoint: str,
        api_key: str | None = None,
        api_version: str = "2024-02-01",
        dimensions: int | None = None,
    ) -> None:
        from openai import AzureOpenAI

        self._deployment = deployment_name
        self._endpoint = endpoint
        self._api_version = api_version

        resolved_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Azure OpenAI API key must be provided via api_key param "
                "or AZURE_OPENAI_API_KEY env var"
            )

        self._client = AzureOpenAI(
            api_key=resolved_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )

        # Default to 1536 if not specified (common for text-embedding-3-small deployments)
        self._dimensions = dimensions or 1536
        self._custom_dimensions = dimensions

        logger.info(
            "azure_openai_embeddings_initialized",
            deployment=self._deployment,
            endpoint=self._endpoint,
            dimensions=self._dimensions,
        )

    def embed_query(self, text: str) -> list[float]:
        result = self._call_api([text])
        return result[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), _MAX_BATCH_SIZE):
            batch = texts[i : i + _MAX_BATCH_SIZE]
            logger.debug(
                "azure_embed_batch",
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
            "provider": "azure_openai",
            "deployment": self._deployment,
            "endpoint": self._endpoint,
            "api_version": self._api_version,
            "dimensions": self._dimensions,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {
            "input": texts,
            "model": self._deployment,
        }
        if self._custom_dimensions is not None:
            kwargs["dimensions"] = self._custom_dimensions

        response = self._client.embeddings.create(**kwargs)
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [d.embedding for d in sorted_data]
