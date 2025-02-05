"""Google Cloud Vertex AI LLM provider implementation."""

from __future__ import annotations

import os
import time
from collections.abc import AsyncGenerator
from typing import Any

import structlog

from .base import BaseLLMProvider, LLMResponse, TokenUsage

logger = structlog.get_logger(__name__)

MODEL_INFO: dict[str, dict[str, Any]] = {
    "gemini-1.5-pro": {
        "max_context": 1_000_000,
        "cost_per_1k_input": 0.00125,
        "cost_per_1k_output": 0.005,
    },
    "gemini-1.5-flash": {
        "max_context": 1_000_000,
        "cost_per_1k_input": 0.000075,
        "cost_per_1k_output": 0.0003,
    },
}


class VertexAIProvider(BaseLLMProvider):
    """Google Cloud Vertex AI provider using the generativeai SDK."""

    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        project: str | None = None,
        location: str = "us-central1",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> None:
        import vertexai
        from vertexai.generative_models import GenerativeModel

        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        resolved_project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not resolved_project:
            raise ValueError("GCP project required. Pass project or set GOOGLE_CLOUD_PROJECT.")

        vertexai.init(project=resolved_project, location=location)
        self._model = GenerativeModel(model)
        logger.info(
            "vertex_provider_initialized",
            model=model,
            project=resolved_project,
            location=location,
        )

    async def generate(
        self,
        prompt: str,
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        from vertexai.generative_models import GenerationConfig

        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        full_prompt = prompt
        if system_message:
            full_prompt = f"{system_message}\n\n{prompt}"

        config = GenerationConfig(
            temperature=temp,
            max_output_tokens=tokens,
        )

        start = time.perf_counter()
        response = self._model.generate_content(full_prompt, generation_config=config)
        latency_ms = (time.perf_counter() - start) * 1000

        usage_meta = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0

        return LLMResponse(
            content=response.text,
            model=self.model_name,
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            latency_ms=latency_ms,
        )

    async def stream(
        self,
        prompt: str,
        system_message: str | None = None,
    ) -> AsyncGenerator[str, None]:
        full_prompt = prompt
        if system_message:
            full_prompt = f"{system_message}\n\n{prompt}"

        response = self._model.generate_content(full_prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text

    def get_model_info(self) -> dict[str, Any]:
        info = MODEL_INFO.get(
            self.model_name,
            {
                "max_context": 1_000_000,
                "cost_per_1k_input": 0.0,
                "cost_per_1k_output": 0.0,
            },
        )
        return {
            "name": self.model_name,
            "provider": "vertex",
            **info,
        }
