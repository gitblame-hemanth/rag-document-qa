"""OpenAI LLM provider implementation."""

import os
import time
from collections.abc import AsyncGenerator
from typing import Any

import structlog

from .base import BaseLLMProvider, LLMResponse, TokenUsage

logger = structlog.get_logger(__name__)

# Model metadata: max_context, cost_per_1k_input, cost_per_1k_output
MODEL_INFO: dict[str, dict[str, Any]] = {
    "gpt-4o": {
        "max_context": 128_000,
        "cost_per_1k_input": 0.0025,
        "cost_per_1k_output": 0.01,
    },
    "gpt-4-turbo": {
        "max_context": 128_000,
        "cost_per_1k_input": 0.01,
        "cost_per_1k_output": 0.03,
    },
    "gpt-4": {
        "max_context": 8_192,
        "cost_per_1k_input": 0.03,
        "cost_per_1k_output": 0.06,
    },
    "gpt-3.5-turbo": {
        "max_context": 16_385,
        "cost_per_1k_input": 0.0005,
        "cost_per_1k_output": 0.0015,
    },
}


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider with retry and exponential backoff."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> None:
        import openai

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("OpenAI API key required. Pass api_key or set OPENAI_API_KEY env var.")

        self._client = openai.AsyncOpenAI(api_key=resolved_key)
        logger.info("openai_provider_initialized", model=model)

    async def generate(
        self,
        prompt: str,
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        import openai

        messages: list[dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                start = time.perf_counter()
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temp,
                    max_tokens=tokens,
                )
                latency_ms = (time.perf_counter() - start) * 1000

                choice = response.choices[0]
                usage = response.usage

                return LLMResponse(
                    content=choice.message.content or "",
                    model=response.model,
                    usage=TokenUsage(
                        input_tokens=usage.prompt_tokens if usage else 0,
                        output_tokens=usage.completion_tokens if usage else 0,
                    ),
                    latency_ms=latency_ms,
                )

            except openai.RateLimitError as exc:
                last_exc = exc
                wait = 2**attempt
                logger.warning(
                    "openai_rate_limit",
                    attempt=attempt + 1,
                    wait_seconds=wait,
                    error=str(exc),
                )
                import asyncio

                await asyncio.sleep(wait)

            except openai.APIError as exc:
                last_exc = exc
                if attempt < self.max_retries - 1 and getattr(exc, "status_code", 0) >= 500:
                    wait = 2**attempt
                    logger.warning(
                        "openai_server_error",
                        attempt=attempt + 1,
                        wait_seconds=wait,
                        error=str(exc),
                    )
                    import asyncio

                    await asyncio.sleep(wait)
                else:
                    raise

        raise last_exc  # type: ignore[misc]

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
            model=self.model,
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
        info = MODEL_INFO.get(
            self.model,
            {
                "max_context": 128_000,
                "cost_per_1k_input": 0.0,
                "cost_per_1k_output": 0.0,
            },
        )
        return {
            "name": self.model,
            "provider": "openai",
            **info,
        }
