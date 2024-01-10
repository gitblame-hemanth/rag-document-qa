"""FastAPI routes for ingest, query, documents, and health."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse

from src.api.models import (
    DocumentInfo,
    ErrorResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceInfo,
    TimingInfo,
)

logger = structlog.get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_chain(request: Request) -> Any:
    return request.app.state.rag_chain


def _get_vectorstore(request: Request) -> Any:
    return request.app.state.vectorstore


def _get_pipeline(request: Request) -> Any:
    return request.app.state.pipeline


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------


@router.post(
    "/ingest",
    response_model=IngestResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Ingest a document",
)
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
    chunking_strategy: str | None = None,
) -> IngestResponse:
    """Accept a file upload, run the ingestion pipeline, and index chunks."""
    pipeline = _get_pipeline(request)

    suffix = Path(file.filename or "upload").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        t0 = time.perf_counter()

        result = await pipeline.ingest_file(tmp_path, chunking_strategy=chunking_strategy)

        duration = time.perf_counter() - t0

        logger.info(
            "document_ingested",
            filename=file.filename,
            num_chunks=result.num_chunks,
            duration_s=round(duration, 3),
        )

        return IngestResponse(
            document_id=result.document_id,
            filename=file.filename or "unknown",
            num_chunks=result.num_chunks,
            duration_seconds=round(duration, 3),
        )
    except Exception as exc:
        logger.error("ingest_failed", filename=file.filename, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Ask a question",
)
async def query_documents(request: Request, body: QueryRequest) -> Any:
    """Run the RAG chain against indexed documents."""
    chain = _get_chain(request)

    # Override top_k if provided
    if body.top_k is not None:
        chain.config.retrieval.top_k = body.top_k

    if body.stream:
        return StreamingResponse(
            _sse_generator(chain, body.question, body.filters),
            media_type="text/event-stream",
        )

    try:
        result = chain.query(body.question, filters=body.filters)
    except Exception as exc:
        logger.error("query_failed", question=body.question[:80], error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sources = [
        SourceInfo(
            filename=c.filename,
            page_number=c.page_number,
            chunk_preview=c.chunk_content,
            relevance_score=c.relevance_score,
        )
        for c in result.sources
    ]

    return QueryResponse(
        answer=result.answer,
        sources=sources,
        confidence=result.confidence,
        timings=TimingInfo(
            retrieval_ms=result.retrieval_time_ms,
            generation_ms=result.generation_time_ms,
        ),
        model_used=result.model_used,
    )


async def _sse_generator(chain: Any, question: str, filters: dict | None):
    """Yield Server-Sent Events from the streaming RAG chain."""
    try:
        async for token in chain.stream_query(question, filters=filters):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        logger.error("stream_failed", error=str(exc))
        yield f"event: error\ndata: {exc!s}\n\n"


# ---------------------------------------------------------------------------
# GET /documents
# ---------------------------------------------------------------------------


@router.get(
    "/documents",
    response_model=list[DocumentInfo],
    summary="List ingested documents",
)
async def list_documents(request: Request) -> list[DocumentInfo]:
    """Return metadata for every ingested document."""
    vs = _get_vectorstore(request)

    try:
        docs = vs.list_documents()
    except Exception as exc:
        logger.error("list_documents_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return [
        DocumentInfo(
            id=d.get("id", ""),
            filename=d.get("filename", ""),
            num_chunks=d.get("num_chunks", 0),
            ingested_at=d.get("ingested_at"),
            file_type=d.get("file_type"),
        )
        for d in docs
    ]


# ---------------------------------------------------------------------------
# DELETE /documents/{document_id}
# ---------------------------------------------------------------------------


@router.delete(
    "/documents/{document_id}",
    status_code=204,
    response_class=Response,
    summary="Delete a document",
)
async def delete_document(request: Request, document_id: str) -> Response:
    """Remove a document and all its chunks from the vector store."""
    vs = _get_vectorstore(request)

    try:
        vs.delete_document(document_id)
    except Exception as exc:
        logger.error("delete_document_failed", document_id=document_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    logger.info("document_deleted", document_id=document_id)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health_check(request: Request) -> HealthResponse:
    """Check vectorstore connectivity and return document count."""
    vs = _get_vectorstore(request)

    try:
        doc_count = vs.count()
        connected = True
    except Exception:
        doc_count = 0
        connected = False

    return HealthResponse(
        status="healthy" if connected else "degraded",
        vectorstore_connected=connected,
        document_count=doc_count,
    )
