"""Tests for LLM provider implementations."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.base import BaseLLMProvider, LLMResponse, TokenUsage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _mock_openai_response(content: str = "Hello", model: str = "gpt-4o"):
    """Build a mock OpenAI ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = model
    return response


# ---------------------------------------------------------------------------
# test_base_is_abstract
# ---------------------------------------------------------------------------


class TestBaseIsAbstract:
    def test_base_is_abstract(self):
        """BaseLLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLLMProvider()


# ---------------------------------------------------------------------------
# test_openai_provider_generate
# ---------------------------------------------------------------------------


class TestOpenAIProviderGenerate:
    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    @patch("openai.AsyncOpenAI")
    def test_openai_provider_generate(self, mock_async_cls):
        """OpenAIProvider.generate returns an LLMResponse with correct fields."""
        from src.llm.openai_provider import OpenAIProvider

        mock_client = MagicMock()
        mock_async_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response("Test answer")
        )

        provider = OpenAIProvider(model="gpt-4o", api_key="sk-test")
        result = _run(provider.generate("What is 1+1?"))

        assert isinstance(result, LLMResponse)
        assert result.content == "Test answer"
        assert result.model == "gpt-4o"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        assert result.usage.total_tokens == 15


# ---------------------------------------------------------------------------
# test_openai_provider_model_info
# ---------------------------------------------------------------------------


class TestOpenAIProviderModelInfo:
    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    @patch("openai.AsyncOpenAI")
    def test_openai_provider_model_info(self, mock_async_cls):
        """get_model_info returns expected metadata dict."""
        from src.llm.openai_provider import OpenAIProvider

        provider = OpenAIProvider(model="gpt-4o", api_key="sk-test")
        info = provider.get_model_info()

        assert info["name"] == "gpt-4o"
        assert info["provider"] == "openai"
        assert "max_context" in info
        assert "cost_per_1k_input" in info


# ---------------------------------------------------------------------------
# test_bedrock_provider_init
# ---------------------------------------------------------------------------


class TestBedrockProviderInit:
    @patch("boto3.client")
    def test_bedrock_provider_init(self, mock_boto_client):
        """BedrockProvider initialises with boto3 and stores config."""
        from src.llm.bedrock_provider import BedrockProvider

        provider = BedrockProvider(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            region="us-west-2",
        )

        assert provider.model == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert provider.temperature == 0.1
        mock_boto_client.assert_called_once_with("bedrock-runtime", region_name="us-west-2")
        info = provider.get_model_info()
        assert info["provider"] == "bedrock"


# ---------------------------------------------------------------------------
# test_azure_provider_init
# ---------------------------------------------------------------------------


class TestAzureProviderInit:
    @patch("openai.AsyncAzureOpenAI")
    def test_azure_provider_init(self, mock_azure_cls):
        """AzureOpenAIProvider initialises with deployment and endpoint."""
        from src.llm.azure_provider import AzureOpenAIProvider

        provider = AzureOpenAIProvider(
            deployment_name="gpt-4-deploy",
            endpoint="https://test.openai.azure.com/",
            api_key="az-key-123",
        )

        assert provider.deployment_name == "gpt-4-deploy"
        assert provider.temperature == 0.1
        mock_azure_cls.assert_called_once()
        info = provider.get_model_info()
        assert info["provider"] == "azure"
        assert info["name"] == "gpt-4-deploy"


# ---------------------------------------------------------------------------
# test_vertex_provider_init
# ---------------------------------------------------------------------------


class TestVertexProviderInit:
    @patch("vertexai.init")
    @patch("vertexai.generative_models.GenerativeModel")
    def test_vertex_provider_init(self, mock_model_cls, mock_init):
        """VertexAIProvider initialises with project and location."""
        from src.llm.vertex_provider import VertexAIProvider

        provider = VertexAIProvider(
            model="gemini-1.5-pro",
            project="my-project",
            location="us-central1",
        )

        assert provider.model_name == "gemini-1.5-pro"
        mock_init.assert_called_once_with(project="my-project", location="us-central1")
        info = provider.get_model_info()
        assert info["provider"] == "vertex"


# ---------------------------------------------------------------------------
# test_factory_openai
# ---------------------------------------------------------------------------


class TestFactoryOpenAI:
    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    @patch("openai.AsyncOpenAI")
    def test_factory_openai(self, mock_async_cls):
        """get_llm_provider('openai') returns an OpenAIProvider."""
        from src.llm.factory import get_llm_provider
        from src.llm.openai_provider import OpenAIProvider

        config = {"provider": "openai", "model": "gpt-4o", "api_key": "sk-test"}
        provider = get_llm_provider(config)

        assert isinstance(provider, OpenAIProvider)


# ---------------------------------------------------------------------------
# test_factory_invalid_provider_raises
# ---------------------------------------------------------------------------


class TestFactoryInvalidProviderRaises:
    def test_factory_invalid_provider_raises(self):
        """get_llm_provider raises ValueError for unknown providers."""
        from src.llm.factory import get_llm_provider

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm_provider({"provider": "not_a_provider"})


# ---------------------------------------------------------------------------
# Token usage dataclass
# ---------------------------------------------------------------------------


class TestTokenUsage:
    def test_token_usage_total(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150
