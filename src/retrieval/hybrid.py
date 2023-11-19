"""Hybrid retrieval — dense vector search combined with BM25 sparse search."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import structlog

from src.retrieval.vectorstore import BaseVectorStore, EmbeddingProvider, SearchResult

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# BM25 sparse index
# ---------------------------------------------------------------------------


class BM25Index:
    """Okapi BM25 index built from a list of chunk texts.

    Parameters
    ----------
    k1 : float
        Term frequency saturation parameter (default 1.5).
    b : float
        Length normalisation parameter (default 0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

        self._corpus: list[list[str]] = []
        self._chunk_ids: list[str] = []
        self._chunk_contents: list[str] = []
        self._chunk_metadatas: list[dict[str, Any]] = []

        # IDF components
        self._doc_count: int = 0
        self._avg_dl: float = 0.0
        self._df: dict[str, int] = defaultdict(int)  # document frequency per term
        self._doc_lens: list[int] = []

    # -- build ---------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase whitespace tokenizer with basic punctuation stripping."""
        import re

        return re.findall(r"\w+", text.lower())

    def build(
        self,
        texts: list[str],
        chunk_ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Build the index from raw chunk texts."""
        self._chunk_contents = texts
        self._chunk_ids = chunk_ids
        self._chunk_metadatas = metadatas or [{} for _ in texts]
        self._doc_count = len(texts)

        self._corpus = []
        self._doc_lens = []
        self._df = defaultdict(int)

        for text in texts:
            tokens = self._tokenize(text)
            self._corpus.append(tokens)
            self._doc_lens.append(len(tokens))
            seen: set[str] = set()
            for t in tokens:
                if t not in seen:
                    self._df[t] += 1
                    seen.add(t)

        self._avg_dl = sum(self._doc_lens) / max(self._doc_count, 1)
        logger.info("bm25_index.build", num_docs=self._doc_count, vocab_size=len(self._df))

    # -- search --------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Score every document against *query* and return top-k results."""
        query_tokens = self._tokenize(query)
        scores: list[float] = [0.0] * self._doc_count

        for token in query_tokens:
            if token not in self._df:
                continue
            df = self._df[token]
            idf = math.log((self._doc_count - df + 0.5) / (df + 0.5) + 1.0)

            for idx, doc_tokens in enumerate(self._corpus):
                tf = doc_tokens.count(token)
                dl = self._doc_lens[idx]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
                scores[idx] += idf * (numerator / denominator)

        ranked = sorted(range(self._doc_count), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            SearchResult(
                content=self._chunk_contents[i],
                metadata=self._chunk_metadatas[i],
                score=scores[i],
                chunk_id=self._chunk_ids[i],
            )
            for i in ranked
            if scores[i] > 0.0
        ]


# ---------------------------------------------------------------------------
# Fusion helpers
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    dense_results: list[SearchResult],
    sparse_results: list[SearchResult],
    k: int = 60,
) -> list[SearchResult]:
    """Merge two ranked lists using Reciprocal Rank Fusion.

    RRF score for document *d*: sum over lists of 1 / (k + rank(d)).
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    result_map: dict[str, SearchResult] = {}

    for rank, r in enumerate(dense_results):
        rrf_scores[r.chunk_id] += 1.0 / (k + rank + 1)
        result_map[r.chunk_id] = r

    for rank, r in enumerate(sparse_results):
        rrf_scores[r.chunk_id] += 1.0 / (k + rank + 1)
        if r.chunk_id not in result_map:
            result_map[r.chunk_id] = r

    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
    return [
        SearchResult(
            content=result_map[cid].content,
            metadata=result_map[cid].metadata,
            score=rrf_scores[cid],
            chunk_id=cid,
        )
        for cid in sorted_ids
    ]


def weighted_combination(
    dense_results: list[SearchResult],
    sparse_results: list[SearchResult],
    alpha: float = 0.7,
) -> list[SearchResult]:
    """Merge by weighted score combination.

    final_score = alpha * dense_score + (1 - alpha) * sparse_score

    Scores are min-max normalised per list before combining.
    """

    def _normalise(results: list[SearchResult]) -> dict[str, float]:
        if not results:
            return {}
        scores = [r.score for r in results]
        lo, hi = min(scores), max(scores)
        span = hi - lo if hi != lo else 1.0
        return {r.chunk_id: (r.score - lo) / span for r in results}

    dense_norm = _normalise(dense_results)
    sparse_norm = _normalise(sparse_results)

    result_map: dict[str, SearchResult] = {}
    for r in dense_results:
        result_map[r.chunk_id] = r
    for r in sparse_results:
        if r.chunk_id not in result_map:
            result_map[r.chunk_id] = r

    all_ids = set(dense_norm) | set(sparse_norm)
    combined: dict[str, float] = {}
    for cid in all_ids:
        d_score = dense_norm.get(cid, 0.0)
        s_score = sparse_norm.get(cid, 0.0)
        combined[cid] = alpha * d_score + (1.0 - alpha) * s_score

    sorted_ids = sorted(combined, key=lambda cid: combined[cid], reverse=True)
    return [
        SearchResult(
            content=result_map[cid].content,
            metadata=result_map[cid].metadata,
            score=combined[cid],
            chunk_id=cid,
        )
        for cid in sorted_ids
    ]


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Combines dense vector search with BM25 sparse search.

    Parameters
    ----------
    vectorstore : BaseVectorStore
        Dense retrieval backend.
    embedder : EmbeddingProvider
        Used to embed the query for dense search.
    alpha : float
        Weight for dense scores in weighted combination (0.0 = pure sparse,
        1.0 = pure dense).  Default 0.7.
    fusion : str
        ``"rrf"`` for reciprocal-rank fusion, ``"weighted"`` for weighted
        score combination.
    """

    def __init__(
        self,
        vectorstore: BaseVectorStore,
        embedder: EmbeddingProvider,
        alpha: float = 0.7,
        fusion: str = "rrf",
    ) -> None:
        self._vectorstore = vectorstore
        self._embedder = embedder
        self._alpha = alpha
        self._fusion = fusion
        self._bm25 = BM25Index()
        self._index_built = False

    # -- index management ----------------------------------------------------

    def build_bm25_index(
        self,
        texts: list[str],
        chunk_ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Build (or rebuild) the BM25 index from chunk texts."""
        self._bm25.build(texts, chunk_ids, metadatas)
        self._index_built = True
        logger.info("hybrid_retriever.bm25_index_built", num_chunks=len(texts))

    # -- search --------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Run hybrid search and return fused results."""
        if not self._index_built:
            logger.warning("hybrid_retriever.bm25_not_built", msg="falling back to dense only")
            query_emb = self._embedder.embed_query(query)
            return self._vectorstore.similarity_search(query_emb, top_k=top_k, filters=filters)

        # Dense search
        query_embedding = self._embedder.embed_query(query)
        dense_results = self._vectorstore.similarity_search(
            query_embedding,
            top_k=top_k * 2,
            filters=filters,
        )

        # Sparse search
        sparse_results = self._bm25.search(query, top_k=top_k * 2)

        # Fuse
        if self._fusion == "rrf":
            merged = reciprocal_rank_fusion(dense_results, sparse_results)
        elif self._fusion == "weighted":
            merged = weighted_combination(dense_results, sparse_results, alpha=self._alpha)
        else:
            raise ValueError(f"Unknown fusion strategy: {self._fusion}")

        logger.info(
            "hybrid_retriever.search",
            query_len=len(query),
            dense_hits=len(dense_results),
            sparse_hits=len(sparse_results),
            fused_hits=len(merged),
            returned=min(top_k, len(merged)),
        )
        return merged[:top_k]
