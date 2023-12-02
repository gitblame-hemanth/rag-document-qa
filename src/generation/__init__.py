"""Generation module — prompts, RAG chain, and LLM providers."""

from src.generation.chain import (
    Citation,
    QueryResult,
    RAGChain,
    get_bedrock_llm,
    get_llm,
    get_openai_llm,
)
from src.generation.prompts import (
    CONDENSE_QUESTION_PROMPT,
    QA_PROMPT,
    SYSTEM_MESSAGE_BEDROCK,
    SYSTEM_MESSAGE_OPENAI,
    format_context,
)

__all__ = [
    "CONDENSE_QUESTION_PROMPT",
    "QA_PROMPT",
    "SYSTEM_MESSAGE_BEDROCK",
    "SYSTEM_MESSAGE_OPENAI",
    "Citation",
    "QueryResult",
    "RAGChain",
    "format_context",
    "get_bedrock_llm",
    "get_llm",
    "get_openai_llm",
]
