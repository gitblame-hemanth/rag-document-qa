"""HuggingFace local embedding provider using sentence-transformers."""

from __future__ import annotations

from typing import Any

import structlog

from embeddings.base import BaseEmbeddingProvider

logger = structlog.get_logger(__name__)


class HuggingFaceEmbeddings(BaseEmbeddingProvider):
    """Embedding provider using sentence-transformers locally.

    No API calls — runs inference on the local machine with optional GPU
    acceleration when CUDA is available.
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model
        self._model: Any = None  # lazy-loaded SentenceTransformer
        self._dimensions: int | None = None
        self._device: str | None = None

        logger.info("hf_embeddings_created", model=self._model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_query(self, text: str) -> list[float]:
        model = self._load_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    def get_dimensions(self) -> int:
        model = self._load_model()
        assert self._dimensions is not None
        return self._dimensions

    def get_model_info(self) -> dict[str, Any]:
        self._load_model()
        return {
            "provider": "huggingface",
            "model": self._model_name,
            "dimensions": self._dimensions,
            "device": self._device,
            "local": True,
        }

    # ------------------------------------------------------------------
    # Internal — lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for HuggingFaceEmbeddings. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        device = self._resolve_device()
        self._device = device

        logger.info(
            "hf_model_loading",
            model=self._model_name,
            device=device,
        )

        self._model = SentenceTransformer(self._model_name, device=device)
        self._dimensions = self._model.get_sentence_embedding_dimension()

        logger.info(
            "hf_model_loaded",
            model=self._model_name,
            dimensions=self._dimensions,
            device=device,
        )
        return self._model

    @staticmethod
    def _resolve_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
