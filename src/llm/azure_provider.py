"""Azure OpenAI LLM provider implementation."""

from __future__ import annotations

import os
import time
from collections.abc import AsyncGenerator
from typing import Any

import structlog

from .base import BaseLLMProvider, LLMResponse, TokenUsage

logger = structlog.get_logger(__name__)


class AzureOpenAIProvider(BaseLLMProvider):
    """Azure OpenAI Service provider."""

    def __init__(
        self,
        deployment_name: str = "gpt-4-deployment",
        endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str = "2024-02-15-preview",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> None:
        import openai

        self.deployment_name = deployment_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        resolved_endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        resolved_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")

        if not resolved_endpoint:
            raise ValueError("Azure endpoint required. Pass endpoint or set AZURE_OPENAI_ENDPOINT.")
        if not resolved_key:
            raise ValueError("Azure API key required. Pass api_key or set AZURE_OPENAI_API_KEY.")

        self._client = openai.AsyncAzureOpenAI(
            azure_endpoint=resolved_endpoint,
            api_key=resolved_key,
            api_version=api_version,
        )
        logger.info(
            "azure_provider_initialized",
            deployment=deployment_name,
            endpoint=resolved_endpoint,
        )

    async def generate(
        self,
        prompt: str,
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        start = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,  # type: ignore[arg-type]
            temperature=temp,
            max_tokens=tokens,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=self.deployment_name,
            usage=TokenUsage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            ),
            latency_ms=latency_ms,
        )

    async def stream(
        self,
        prompt: str,
        system_message: str | None = None,
    ) -> AsyncGenerator[str, None]:
        messages: list[dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    def get_model_info(self) -> dict[str, Any]:
        return {
            "name": self.deployment_name,
            "provider": "azure",
            "max_context": 128_000,
            "cost_per_1k_input": 0.0,
            "cost_per_1k_output": 0.0,
        }
