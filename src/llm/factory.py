"""LLM provider factory — routes config to the correct provider implementation."""

from typing import Any

import structlog

from .base import BaseLLMProvider

logger = structlog.get_logger(__name__)


def get_llm_provider(config: Any) -> BaseLLMProvider:
    """Create and return an LLM provider based on application config.

    Expects ``config.llm.provider`` to be one of:
        openai, azure_openai, bedrock, vertex_ai

    Additional keys under ``config.llm`` are forwarded as kwargs to the
    provider constructor (e.g. ``config.llm.model``, ``config.llm.api_key``).

    Args:
        config: Application configuration object with an ``llm`` namespace.

    Returns:
        An initialised BaseLLMProvider subclass.

    Raises:
        ValueError: If the provider string is not recognised.
        ImportError: If the required SDK for the provider is not installed.
    """
    llm_cfg = config.llm if hasattr(config, "llm") else config
    provider_name: str = getattr(llm_cfg, "provider", None) or llm_cfg.get("provider", "")
    provider_name = provider_name.strip().lower()

    logger.info("llm_factory_creating_provider", provider=provider_name)

    # ---------- OpenAI ----------
    if provider_name == "openai":
        try:
            from .openai_provider import OpenAIProvider
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for the OpenAI provider. "
                "Install it with: pip install openai"
            )
        return OpenAIProvider(
            model=_get(llm_cfg, "model", "gpt-4o"),
            api_key=_get(llm_cfg, "api_key", None),
            temperature=_get(llm_cfg, "temperature", 0.1),
            max_tokens=_get(llm_cfg, "max_tokens", 4096),
        )

    # ---------- Azure OpenAI ----------
    if provider_name in ("azure_openai", "azure-openai", "azure"):
        try:
            from .azure_provider import AzureOpenAIProvider
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for the Azure OpenAI provider. "
                "Install it with: pip install openai"
            )
        deployment = _get(llm_cfg, "deployment_name", None) or _get(llm_cfg, "model", None)
        if not deployment:
            raise ValueError("Azure OpenAI requires 'deployment_name' or 'model' in config.")
        return AzureOpenAIProvider(
            deployment_name=deployment,
            endpoint=_get(llm_cfg, "endpoint", None),
            api_key=_get(llm_cfg, "api_key", None),
            api_version=_get(llm_cfg, "api_version", "2024-02-01"),
            temperature=_get(llm_cfg, "temperature", 0.1),
            max_tokens=_get(llm_cfg, "max_tokens", 4096),
        )

    # ---------- AWS Bedrock ----------
    if provider_name == "bedrock":
        try:
            from .bedrock_provider import BedrockProvider
        except ImportError:
            raise ImportError(
                "The 'boto3' package is required for the Bedrock provider. "
                "Install it with: pip install boto3"
            )
        return BedrockProvider(
            model=_get(llm_cfg, "model", "anthropic.claude-3-sonnet"),
            region=_get(llm_cfg, "region", None),
            temperature=_get(llm_cfg, "temperature", 0.1),
            max_tokens=_get(llm_cfg, "max_tokens", 4096),
        )

    # ---------- GCP Vertex AI ----------
    if provider_name in ("vertex_ai", "vertex-ai", "vertex", "vertexai"):
        try:
            from .vertex_provider import VertexAIProvider
        except ImportError:
            raise ImportError(
                "The 'google-cloud-aiplatform' package is required for the Vertex AI provider. "
                "Install it with: pip install google-cloud-aiplatform"
            )
        return VertexAIProvider(
            model=_get(llm_cfg, "model", "gemini-1.5-pro"),
            project=_get(llm_cfg, "project", None),
            location=_get(llm_cfg, "location", "us-central1"),
            temperature=_get(llm_cfg, "temperature", 0.1),
            max_tokens=_get(llm_cfg, "max_tokens", 4096),
        )

    raise ValueError(
        f"Unknown LLM provider: '{provider_name}'. "
        f"Supported providers: openai, azure_openai, bedrock, vertex_ai"
    )


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    """Extract a value from config, supporting both attribute and dict access."""
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return default
