"""AWS Bedrock embedding provider."""

from __future__ import annotations

import json
import os
from typing import Any

import structlog

from embeddings.base import BaseEmbeddingProvider

logger = structlog.get_logger(__name__)

_MODEL_DIMENSIONS: dict[str, int] = {
    "amazon.titan-embed-text-v2:0": 1024,
    "amazon.titan-embed-text-v1": 1536,
    "cohere.embed-english-v3": 1024,
    "cohere.embed-multilingual-v3": 1024,
}


class BedrockEmbeddings(BaseEmbeddingProvider):
    """Embedding provider using AWS Bedrock runtime."""

    def __init__(
        self,
        model: str = "amazon.titan-embed-text-v2:0",
        region: str | None = None,
    ) -> None:
        import boto3

        self._model = model
        resolved_region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

        self._client = boto3.client("bedrock-runtime", region_name=resolved_region)

        if model not in _MODEL_DIMENSIONS:
            raise ValueError(f"Unsupported model '{model}'. Supported: {list(_MODEL_DIMENSIONS)}")

        self._dimensions = _MODEL_DIMENSIONS[model]
        self._region = resolved_region

        logger.info(
            "bedrock_embeddings_initialized",
            model=self._model,
            region=self._region,
            dimensions=self._dimensions,
        )

    def embed_query(self, text: str) -> list[float]:
        return self._invoke(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float]] = []
        for idx, text in enumerate(texts):
            logger.debug(
                "bedrock_embed_document",
                index=idx,
                total=len(texts),
            )
            results.append(self._invoke(text))
        return results

    def get_dimensions(self) -> int:
        return self._dimensions

    def get_model_info(self) -> dict[str, Any]:
        return {
            "provider": "bedrock",
            "model": self._model,
            "region": self._region,
            "dimensions": self._dimensions,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _invoke(self, text: str) -> list[float]:
        body = self._build_request_body(text)
        response = self._client.invoke_model(
            modelId=self._model,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        response_body = json.loads(response["body"].read())
        return self._parse_response(response_body)

    def _build_request_body(self, text: str) -> dict[str, Any]:
        if self._model.startswith("amazon.titan"):
            return {"inputText": text}
        if self._model.startswith("cohere."):
            return {
                "texts": [text],
                "input_type": "search_document",
                "truncate": "END",
            }
        raise ValueError(f"Unknown model family for '{self._model}'")

    def _parse_response(self, body: dict[str, Any]) -> list[float]:
        if self._model.startswith("amazon.titan"):
            return body["embedding"]
        if self._model.startswith("cohere."):
            return body["embeddings"][0]
        raise ValueError(f"Unknown model family for '{self._model}'")
