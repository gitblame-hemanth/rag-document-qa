"""Pydantic request/response models for the RAG API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    """Optional metadata sent alongside the file upload."""

    chunking_strategy: str | None = Field(
        default=None,
        description="Override chunking strategy: 'fixed', 'recursive', or 'semantic'.",
    )


class QueryRequest(BaseModel):
    """JSON body for the /query endpoint."""

    question: str = Field(..., min_length=1, description="The question to answer.")
    top_k: int | None = Field(
        default=None, ge=1, le=50, description="Number of chunks to retrieve."
    )
    filters: dict[str, Any] | None = Field(
        default=None, description="Metadata filters for retrieval (e.g. filename, file_type)."
    )
    stream: bool = Field(default=False, description="If true, return SSE streaming response.")


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class IngestResponse(BaseModel):
    """Response after successful document ingestion."""

    document_id: str
    filename: str
    num_chunks: int
    duration_seconds: float


class SourceInfo(BaseModel):
    """A single cited source in a query response."""

    filename: str
    page_number: int | str
    chunk_preview: str
    relevance_score: float


class TimingInfo(BaseModel):
    """Latency breakdown for a query."""

    retrieval_ms: float
    generation_ms: float


class QueryResponse(BaseModel):
    """Response from the /query endpoint."""

    answer: str
    sources: list[SourceInfo]
    confidence: float
    timings: TimingInfo
    model_used: str


class DocumentInfo(BaseModel):
    """Metadata for a single ingested document."""

    id: str
    filename: str
    num_chunks: int
    ingested_at: datetime | None = None
    file_type: str | None = None


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str
    vectorstore_connected: bool
    document_count: int


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    detail: str
    request_id: str | None = None
