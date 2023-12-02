"""Prompt templates for RAG generation for RAG generation."""

from __future__ import annotations

SYSTEM_MESSAGE_OPENAI = (
    "You are a precise document question-answering assistant. "
    "You answer questions based ONLY on the provided context from the user's documents. "
    "You always cite your sources using [Source: filename, Page: N] format. "
    "If the context does not contain enough information to answer the question, "
    "you respond with: "
    '"I don\'t have enough information in the provided documents to answer this question."'
)

SYSTEM_MESSAGE_BEDROCK = (
    "\n\nHuman: You are a precise document question-answering assistant. "
    "You answer questions based ONLY on the provided context from the user's documents. "
    "You always cite your sources using [Source: filename, Page: N] format. "
    "If the context does not contain enough information to answer the question, "
    "you respond with: "
    '"I don\'t have enough information in the provided documents to answer this question."\n\n'
)

QA_PROMPT = """Use the following context from retrieved documents to answer the question.
Rules:
1. Answer ONLY based on the provided context. Do NOT use prior knowledge.
2. Cite every claim with [Source: filename, Page: N] using the metadata below each chunk.
3. If the context does not contain sufficient information, respond exactly with:
   "I don't have enough information in the provided documents to answer this question."
4. Be concise and direct.

--- CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}

Answer:"""

CONDENSE_QUESTION_PROMPT = """Given the following conversation history and a follow-up question,
rephrase the follow-up question to be a standalone question that captures the full intent.
Do NOT answer the question — only rephrase it.

Chat History:
{chat_history}

Follow-Up Question: {question}

Standalone Question:"""


def format_context(search_results: list) -> str:
    """Format search results into a context string with source citations.

    Each result is expected to have content, filename, page_number,
    and relevance_score attributes (matching SearchResult from retrieval).
    """
    if not search_results:
        return "No relevant documents found."

    parts: list[str] = []
    for i, result in enumerate(search_results, 1):
        filename = getattr(result, "filename", None) or result.metadata.get("filename", "unknown")
        page = getattr(result, "page_number", None) or result.metadata.get("page_number", "N/A")
        score = getattr(result, "relevance_score", None) or result.metadata.get(
            "relevance_score", 0.0
        )
        content = getattr(result, "content", None) or getattr(result, "page_content", "")

        parts.append(
            f"[Chunk {i}]\nSource: {filename} | Page: {page} | Relevance: {score:.3f}\n{content}\n"
        )

    return "\n".join(parts)
