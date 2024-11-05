"""Abstract base class for LLM providers and shared data structures."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TokenUsage:
    """Token usage statistics for an LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""

    content: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0


class BaseLLMProvider(ABC):
    """Abstract base class that all LLM providers must implement."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a completion from the LLM.

        Args:
            prompt: The user prompt / query.
            system_message: Optional system-level instruction.
            temperature: Sampling temperature override.
            max_tokens: Max output tokens override.

        Returns:
            LLMResponse with content, model info, usage, and latency.
        """
        ...

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        system_message: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from the LLM.

        Args:
            prompt: The user prompt / query.
            system_message: Optional system-level instruction.

        Yields:
            Individual text chunks as they arrive.
        """
        ...
        # Needed so Python treats this as an async generator
        yield  # pragma: no cover

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Return metadata about the current model configuration.

        Returns:
            Dict with keys: name, provider, max_context,
            cost_per_1k_input, cost_per_1k_output.
        """
        ...
