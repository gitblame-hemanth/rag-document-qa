"""LLM provider abstraction layer.

Exports all provider classes, shared data structures, and the factory function.
"""

from .base import BaseLLMProvider, LLMResponse, TokenUsage
from .factory import get_llm_provider

__all__ = [
    "BaseLLMProvider",
    "LLMResponse",
    "TokenUsage",
    "get_llm_provider",
    # Provider classes — imported lazily by the factory, but re-exported
    # here for direct use.
    "OpenAIProvider",
    "AzureOpenAIProvider",
    "BedrockProvider",
    "VertexAIProvider",
]


def __getattr__(name: str):
    """Lazy-load provider classes to avoid hard dependency on every SDK."""
    if name == "OpenAIProvider":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider
    if name == "AzureOpenAIProvider":
        from .azure_provider import AzureOpenAIProvider

        return AzureOpenAIProvider
    if name == "BedrockProvider":
        from .bedrock_provider import BedrockProvider

        return BedrockProvider
    if name == "VertexAIProvider":
        from .vertex_provider import VertexAIProvider

        return VertexAIProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
