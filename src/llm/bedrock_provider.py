"""AWS Bedrock LLM provider implementation."""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncGenerator
from typing import Any

import structlog

from .base import BaseLLMProvider, LLMResponse, TokenUsage

logger = structlog.get_logger(__name__)

MODEL_INFO: dict[str, dict[str, Any]] = {
    "anthropic.claude-3-sonnet-20240229-v1:0": {
        "max_context": 200_000,
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
    },
    "anthropic.claude-3-haiku-20240307-v1:0": {
        "max_context": 200_000,
        "cost_per_1k_input": 0.00025,
        "cost_per_1k_output": 0.00125,
    },
}


class BedrockProvider(BaseLLMProvider):
    """AWS Bedrock provider using the boto3 runtime client."""

    def __init__(
        self,
        model: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        region: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> None:
        import boto3

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self._client = boto3.client("bedrock-runtime", region_name=region)
        logger.info("bedrock_provider_initialized", model=model, region=region)

    async def generate(
        self,
        prompt: str,
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        messages = [{"role": "user", "content": prompt}]

        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": tokens,
            "temperature": temp,
            "messages": messages,
        }
        if system_message:
            body["system"] = system_message

        start = time.perf_counter()
        response = self._client.invoke_model(
            modelId=self.model,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        latency_ms = (time.perf_counter() - start) * 1000

        result = json.loads(response["body"].read())
        content = result.get("content", [{}])[0].get("text", "")
        usage = result.get("usage", {})

        return LLMResponse(
            content=content,
            model=self.model,
            usage=TokenUsage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            ),
            latency_ms=latency_ms,
        )

    async def stream(
        self,
        prompt: str,
        system_message: str | None = None,
    ) -> AsyncGenerator[str, None]:
        messages = [{"role": "user", "content": prompt}]

        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
        }
        if system_message:
            body["system"] = system_message

        response = self._client.invoke_model_with_response_stream(
            modelId=self.model,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        for event in response.get("body", []):
            chunk = json.loads(event.get("chunk", {}).get("bytes", b"{}"))
            if chunk.get("type") == "content_block_delta":
                delta = chunk.get("delta", {}).get("text", "")
                if delta:
                    yield delta

    def get_model_info(self) -> dict[str, Any]:
        info = MODEL_INFO.get(
            self.model,
            {
                "max_context": 200_000,
                "cost_per_1k_input": 0.0,
                "cost_per_1k_output": 0.0,
            },
        )
        return {
            "name": self.model,
            "provider": "bedrock",
            **info,
        }
