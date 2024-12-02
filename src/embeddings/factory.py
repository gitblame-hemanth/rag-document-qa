"""Factory for creating embedding providers from configuration."""

from __future__ import annotations

from typing import Any

import structlog

from embeddings.base import BaseEmbeddingProvider

logger = structlog.get_logger(__name__)


def get_embedding_provider(config: Any) -> BaseEmbeddingProvider:
    """Instantiate an embedding provider based on ``config.embeddings.provider``.

    Args:
        config: Application configuration object. Must expose an ``embeddings``
            attribute (or subscriptable key) with at least a ``provider`` field.

    Returns:
        A fully initialised :class:`BaseEmbeddingProvider`.

    Raises:
        ValueError: If the provider name is unrecognised.
    """
    emb_cfg = _extract_embeddings_config(config)
    provider = emb_cfg.get("provider", "openai")

    logger.info("embedding_provider_requested", provider=provider)

    if provider == "openai":
        from embeddings.openai_embed import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=emb_cfg.get("model", "text-embedding-3-small"),
            api_key=emb_cfg.get("api_key"),
            dimensions=emb_cfg.get("dimensions"),
        )

    if provider == "bedrock":
        from embeddings.bedrock_embed import BedrockEmbeddings

        return BedrockEmbeddings(
            model=emb_cfg.get("model", "amazon.titan-embed-text-v2:0"),
            region=emb_cfg.get("region"),
        )

    if provider == "azure_openai":
        from embeddings.azure_embed import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings(
            deployment_name=emb_cfg["deployment_name"],
            endpoint=emb_cfg["endpoint"],
            api_key=emb_cfg.get("api_key"),
            api_version=emb_cfg.get("api_version", "2024-02-01"),
            dimensions=emb_cfg.get("dimensions"),
        )

    if provider == "huggingface":
        from embeddings.hf_embed import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model=emb_cfg.get("model", "all-MiniLM-L6-v2"),
        )

    raise ValueError(
        f"Unknown embedding provider '{provider}'. "
        "Supported: openai, bedrock, azure_openai, huggingface"
    )


def _extract_embeddings_config(config: Any) -> dict[str, Any]:
    """Normalise config into a plain dict of embedding settings."""
    # Support attribute-style (e.g. dataclass / pydantic)
    if hasattr(config, "embeddings"):
        section = config.embeddings
        if isinstance(section, dict):
            return section
        # Assume object with __dict__
        return vars(section) if hasattr(section, "__dict__") else {"provider": str(section)}

    # Support dict-style
    if isinstance(config, dict):
        return config.get("embeddings", config)

    raise TypeError(
        f"Cannot extract embeddings config from {type(config).__name__}. "
        "Expected an object with an 'embeddings' attribute or a dict."
    )
