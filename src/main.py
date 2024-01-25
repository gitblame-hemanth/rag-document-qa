"""FastAPI application factory and entrypoint."""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import load_config

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — initialise all components once at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: build config, vectorstore, embeddings, retriever, reranker, RAG chain."""
    config = load_config()

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            (
                structlog.dev.ConsoleRenderer()
                if config.logging.format == "console"
                else structlog.processors.JSONRenderer()
            ),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            structlog._log_levels._NAME_TO_LEVEL[config.logging.level.lower()]
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    log = structlog.get_logger("startup")

    # Embeddings
    from src.retrieval import get_embeddings, get_reranker, get_vectorstore

    embeddings = get_embeddings(config.embedding)
    log.info("embeddings_ready", provider=config.embedding.provider)

    # Vector store
    vectorstore = get_vectorstore(config.vectorstore)
    log.info("vectorstore_ready", provider=config.vectorstore.provider)

    # Retriever (hybrid if enabled, else None — chain falls back to vectorstore)
    retriever = None
    if config.retrieval.search_type == "hybrid":
        from src.retrieval import HybridRetriever

        retriever = HybridRetriever(vectorstore=vectorstore, config=config.retrieval)
        log.info("hybrid_retriever_ready")

    # Reranker
    reranker = get_reranker(config.retrieval)
    log.info(
        "reranker_ready",
        enabled=config.retrieval.reranker_enabled,
        model=config.retrieval.reranker_model,
    )

    # Ingestion pipeline
    from src.ingestion import IngestPipeline

    pipeline = IngestPipeline(vectorstore=vectorstore, embedding_fn=embeddings.embed_documents)
    log.info("ingestion_pipeline_ready")

    # RAG chain
    from src.generation import RAGChain

    rag_chain = RAGChain(
        config=config,
        vectorstore=vectorstore,
        retriever=retriever,
        reranker=reranker,
    )
    log.info("rag_chain_ready")

    # Attach to app state
    app.state.config = config
    app.state.vectorstore = vectorstore
    app.state.retriever = retriever
    app.state.reranker = reranker
    app.state.pipeline = pipeline
    app.state.rag_chain = rag_chain

    log.info("application_started", host=config.api.host, port=config.api.port)
    yield
    log.info("application_shutdown")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="RAG Document QA",
        description="Retrieval-Augmented Generation API for document question answering.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # -- CORS ---------------------------------------------------------------
    # Actual origins are set in lifespan after config loads; use permissive
    # defaults here so the middleware object exists. The lifespan will update
    # if needed.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # overridden per-config in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- Request-ID middleware -----------------------------------------------
    @application.middleware("http")
    async def request_id_middleware(request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # -- Global exception handler -------------------------------------------
    @application.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_exception", error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "request_id": request.headers.get("X-Request-ID"),
            },
        )

    # -- Routes -------------------------------------------------------------
    from src.api.routes import router

    application.include_router(router)

    return application


# Module-level app for uvicorn / gunicorn
app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
