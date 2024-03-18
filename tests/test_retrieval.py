"""Tests for retrieval module — BM25, RRF, and hybrid search."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.retrieval.vectorstore import SearchResult


def _make_result(content: str, score: float = 0.0, chunk_id: str = "") -> SearchResult:
    return SearchResult(content=content, metadata={}, score=score, chunk_id=chunk_id)


@pytest.fixture()
def corpus() -> list[str]:
    return [
        "Acme Corp was founded in 2010 and specializes in cloud infrastructure.",
        "Employees receive 15 days of paid vacation per year.",
        "The remote work policy allows up to three days per week from home.",
        "All customer data must be encrypted at rest using AES-256.",
        "Performance reviews are conducted semi-annually in January and July.",
    ]


@pytest.fixture()
def chunk_ids() -> list[str]:
    return ["c0", "c1", "c2", "c3", "c4"]


class TestBM25Search:
    def test_bm25_returns_ranked_results(self, corpus, chunk_ids):
        from src.retrieval.hybrid import BM25Index

        idx = BM25Index()
        idx.build(corpus, chunk_ids)

        results = idx.search("vacation days employees", top_k=3)

        assert len(results) <= 3
        # The vacation-related doc should rank highest
        assert any("vacation" in r.content.lower() for r in results)

    def test_bm25_empty_query_returns_empty(self, corpus, chunk_ids):
        from src.retrieval.hybrid import BM25Index

        idx = BM25Index()
        idx.build(corpus, chunk_ids)

        results = idx.search("", top_k=3)
        assert len(results) == 0

    def test_bm25_top_k_limits_results(self, corpus, chunk_ids):
        from src.retrieval.hybrid import BM25Index

        idx = BM25Index()
        idx.build(corpus, chunk_ids)

        results = idx.search("company cloud data", top_k=2)
        assert len(results) <= 2


class TestReciprocalRankFusion:
    def test_rrf_merges_two_ranked_lists(self):
        from src.retrieval.hybrid import reciprocal_rank_fusion

        list_a = [
            _make_result("doc1", score=0.9, chunk_id="1"),
            _make_result("doc2", score=0.8, chunk_id="2"),
            _make_result("doc3", score=0.7, chunk_id="3"),
        ]
        list_b = [
            _make_result("doc2", score=0.95, chunk_id="2"),
            _make_result("doc4", score=0.85, chunk_id="4"),
            _make_result("doc1", score=0.75, chunk_id="1"),
        ]

        fused = reciprocal_rank_fusion(list_a, list_b, k=60)

        contents = [r.content for r in fused]
        # doc1 and doc2 appear in both lists, should rank high
        assert "doc1" in contents
        assert "doc2" in contents
        # All unique docs should be present
        assert len(set(contents)) == 4

    def test_rrf_single_list_preserves_order(self):
        from src.retrieval.hybrid import reciprocal_rank_fusion

        results = [
            _make_result("a", score=0.9, chunk_id="a"),
            _make_result("b", score=0.5, chunk_id="b"),
        ]

        fused = reciprocal_rank_fusion(results, [], k=60)
        assert [r.content for r in fused] == ["a", "b"]

    def test_rrf_empty_lists(self):
        from src.retrieval.hybrid import reciprocal_rank_fusion

        fused = reciprocal_rank_fusion([], [], k=60)
        assert fused == []


class TestHybridSearch:
    def test_hybrid_search_combines_results(self, corpus, chunk_ids):
        """Hybrid search should combine vector and BM25 results."""
        from src.retrieval.hybrid import HybridRetriever

        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search.return_value = [
            _make_result(corpus[0], score=0.95, chunk_id="c0"),
            _make_result(corpus[1], score=0.85, chunk_id="c1"),
        ]

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 128

        retriever = HybridRetriever(
            vectorstore=mock_vectorstore,
            embedder=mock_embedder,
            alpha=0.7,
            fusion="rrf",
        )

        # Build BM25 index from corpus
        retriever.build_bm25_index(corpus, chunk_ids)

        results = retriever.search("vacation policy")

        assert len(results) > 0
        assert all(hasattr(r, "content") for r in results)

    def test_hybrid_retriever_falls_back_to_vector_only(self):
        """When BM25 index is not built, should fall back to vector search only."""
        from src.retrieval.hybrid import HybridRetriever

        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search.return_value = [
            _make_result("fallback result", score=0.9, chunk_id="fb1"),
        ]

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 128

        retriever = HybridRetriever(
            vectorstore=mock_vectorstore,
            embedder=mock_embedder,
        )

        results = retriever.search("vacation policy")

        # Should still return results from vector store
        assert len(results) > 0
        mock_vectorstore.similarity_search.assert_called_once()
