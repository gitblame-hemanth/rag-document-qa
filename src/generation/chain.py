"""RAG chain for document QA — orchestrates retrieval and generation with citation extraction."""

from __future__ import annotations

import re
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import structlog
from langchain_openai import ChatOpenAI

from src.config import AppConfig, LLMConfig
from src.generation.prompts import (
    CONDENSE_QUESTION_PROMPT,
    QA_PROMPT,
    SYSTEM_MESSAGE_BEDROCK,
    SYSTEM_MESSAGE_OPENAI,
    format_context,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Citation:
    """A single source citation extracted from the generated answer."""

    filename: str
    page_number: int | str
    chunk_content: str  # truncated preview
    relevance_score: float


@dataclass
class QueryResult:
    """Full result of a RAG query."""

    answer: str
    sources: list[Citation]
    confidence: float
    retrieval_time_ms: float
    generation_time_ms: float
    model_used: str


# ---------------------------------------------------------------------------
# LLM provider factories
# ---------------------------------------------------------------------------


def get_openai_llm(config: LLMConfig) -> ChatOpenAI:
    """Create a ChatOpenAI instance from config."""
    return ChatOpenAI(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )


def get_bedrock_llm(config: LLMConfig) -> Any:
    """Create a ChatBedrock instance from config.

    Imports lazily to avoid hard dependency on boto3 when using OpenAI.
    """
    from langchain_aws import ChatBedrock  # type: ignore[import-untyped]

    return ChatBedrock(
        model_id=config.model,
        model_kwargs={
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        },
    )


def get_llm(config: LLMConfig) -> Any:
    """Factory — returns the appropriate LLM based on provider setting."""
    providers = {
        "openai": get_openai_llm,
        "bedrock": get_bedrock_llm,
    }
    factory = providers.get(config.provider)
    if factory is None:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")
    return factory(config)


# ---------------------------------------------------------------------------
# RAG chain
# ---------------------------------------------------------------------------


class RAGChain:
    """End-to-end retrieval-augmented generation chain."""

    _CITE_PATTERN = re.compile(r"\[Source:\s*(?P<filename>[^,\]]+),\s*Page:\s*(?P<page>[^\]]+)\]")
    _MAX_CHUNK_PREVIEW = 200

    def __init__(
        self,
        config: AppConfig,
        vectorstore: Any,
        retriever: Any,
        reranker: Any,
    ) -> None:
        self.config = config
        self.vectorstore = vectorstore
        self.retriever = retriever
        self.reranker = reranker
        self.llm = get_llm(config.llm)
        self.chat_history: list[tuple[str, str]] = []

        self._system_msg = (
            SYSTEM_MESSAGE_BEDROCK if config.llm.provider == "bedrock" else SYSTEM_MESSAGE_OPENAI
        )
        logger.info(
            "rag_chain_initialized",
            llm_provider=config.llm.provider,
            llm_model=config.llm.model,
        )

    # -- public API --------------------------------------------------------

    def query(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> QueryResult:
        """Run a full RAG query synchronously."""
        effective_question = self._condense_question(question)

        t0 = time.perf_counter()
        context_results = self._retrieve_context(effective_question, filters)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        prompt = self._build_prompt(effective_question, context_results)

        t1 = time.perf_counter()
        response = self._invoke_llm(prompt)
        generation_ms = (time.perf_counter() - t1) * 1000

        citations = self._extract_citations(response, context_results)
        confidence = self._score_confidence(response, context_results)

        # Update conversation memory
        self.chat_history.append((question, response))

        result = QueryResult(
            answer=response,
            sources=citations,
            confidence=confidence,
            retrieval_time_ms=round(retrieval_ms, 2),
            generation_time_ms=round(generation_ms, 2),
            model_used=self.config.llm.model,
        )

        logger.info(
            "query_completed",
            question=question[:80],
            confidence=confidence,
            num_sources=len(citations),
            retrieval_ms=result.retrieval_time_ms,
            generation_ms=result.generation_time_ms,
        )
        return result

    async def stream_query(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream the generated answer token-by-token via async generator."""
        effective_question = self._condense_question(question)
        context_results = self._retrieve_context(effective_question, filters)
        prompt = self._build_prompt(effective_question, context_results)

        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=self._system_msg),
            HumanMessage(content=prompt),
        ]

        collected: list[str] = []
        async for chunk in self.llm.astream(messages):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            collected.append(token)
            yield token

        full_response = "".join(collected)
        self.chat_history.append((question, full_response))

    # -- internals ---------------------------------------------------------

    def _condense_question(self, question: str) -> str:
        """If there is chat history, condense follow-up into standalone question."""
        if not self.chat_history:
            return question

        history_text = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in self.chat_history[-5:])
        condense_prompt = CONDENSE_QUESTION_PROMPT.format(
            chat_history=history_text, question=question
        )

        from langchain_core.messages import HumanMessage

        response = self.llm.invoke([HumanMessage(content=condense_prompt)])
        condensed = response.content if hasattr(response, "content") else str(response)
        logger.debug("question_condensed", original=question[:60], condensed=condensed[:60])
        return condensed.strip()

    def _retrieve_context(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Retrieve and optionally rerank context chunks."""
        top_k = self.config.retrieval.top_k

        if self.retriever is not None:
            results = self.retriever.search(question, top_k=top_k, filters=filters)
        else:
            results = self.vectorstore.search(
                question,
                top_k=top_k,
                search_type=self.config.retrieval.search_type,
                filters=filters,
            )

        if self.reranker is not None and results:
            results = self.reranker.rerank(question, results)

        logger.debug("context_retrieved", num_chunks=len(results))
        return results

    def _build_prompt(self, question: str, context: list[Any]) -> str:
        """Build the final prompt from context and question."""
        context_str = format_context(context)
        return QA_PROMPT.format(context=context_str, question=question)

    def _invoke_llm(self, prompt: str) -> str:
        """Send prompt to LLM and return the text response."""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=self._system_msg),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    def _extract_citations(self, response: str, context: list[Any]) -> list[Citation]:
        """Extract [Source: ..., Page: ...] citations and match to context chunks."""
        cited: list[Citation] = []
        seen: set[tuple[str, str]] = set()

        for match in self._CITE_PATTERN.finditer(response):
            fname = match.group("filename").strip()
            page = match.group("page").strip()
            key = (fname, page)
            if key in seen:
                continue
            seen.add(key)

            # Find matching context chunk for content preview & score
            chunk_content = ""
            score = 0.0
            for result in context:
                r_fname = getattr(result, "filename", None) or result.metadata.get("filename", "")
                r_page = str(
                    getattr(result, "page_number", None) or result.metadata.get("page_number", "")
                )
                if r_fname == fname and r_page == page:
                    chunk_content = getattr(result, "content", None) or getattr(
                        result, "page_content", ""
                    )
                    score = getattr(result, "relevance_score", 0.0)
                    break

            cited.append(
                Citation(
                    filename=fname,
                    page_number=page,
                    chunk_content=chunk_content[: self._MAX_CHUNK_PREVIEW],
                    relevance_score=round(score, 4),
                )
            )

        return cited

    def _score_confidence(self, response: str, context: list[Any]) -> float:
        """Heuristic confidence score based on source coverage and specificity.

        Factors:
        - Proportion of top-k results that were actually cited (source coverage).
        - Whether the answer is a refusal ("I don't have enough information").
        - Average relevance score of cited chunks.
        """
        if not context:
            return 0.0

        # Refusal detection
        refusal_phrases = [
            "i don't have enough information",
            "not enough information",
            "cannot answer",
            "no relevant",
        ]
        lower_resp = response.lower()
        if any(phrase in lower_resp for phrase in refusal_phrases):
            return 0.0

        # Source coverage: how many context chunks were cited
        cited_matches = self._CITE_PATTERN.findall(response)
        num_cited = len(set(cited_matches))
        coverage = min(num_cited / max(len(context), 1), 1.0)

        # Average relevance of retrieved chunks
        scores = []
        for r in context:
            s = getattr(r, "relevance_score", None)
            if s is None:
                s = r.metadata.get("relevance_score", 0.0) if hasattr(r, "metadata") else 0.0
            scores.append(float(s))
        avg_relevance = sum(scores) / max(len(scores), 1)

        # Answer length heuristic (very short answers are less confident)
        length_factor = min(len(response.split()) / 20, 1.0)

        confidence = (0.4 * coverage) + (0.4 * avg_relevance) + (0.2 * length_factor)
        return round(min(max(confidence, 0.0), 1.0), 3)
