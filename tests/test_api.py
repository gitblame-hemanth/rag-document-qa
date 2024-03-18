"""Tests for FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import router


@pytest.fixture()
def client():
    """Create a TestClient with mocked app state."""
    app = FastAPI()
    app.include_router(router)

    # Mock rag_chain
    mock_chain = MagicMock()
    mock_chain.query.return_value = MagicMock(
        answer="Acme Corp was founded in 2010.",
        sources=[],
        confidence=0.95,
        retrieval_time_ms=10.0,
        generation_time_ms=50.0,
        model_used="test-model",
    )
    app.state.rag_chain = mock_chain

    # Mock vectorstore
    mock_vs = MagicMock()
    mock_vs.list_documents.return_value = [
        {
            "id": "1",
            "filename": "about.txt",
            "num_chunks": 5,
            "ingested_at": None,
            "file_type": "txt",
        },
        {"id": "2", "filename": "hr.txt", "num_chunks": 3, "ingested_at": None, "file_type": "txt"},
    ]
    mock_vs.count.return_value = 2
    app.state.vectorstore = mock_vs

    # Mock pipeline
    app.state.pipeline = MagicMock()

    yield TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_body(self, client: TestClient):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"


class TestQueryEndpoint:
    def test_query_returns_answer(self, client: TestClient):
        response = client.post("/query", json={"question": "When was Acme Corp founded?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "Acme" in data["answer"]

    def test_query_includes_sources(self, client: TestClient):
        response = client.post("/query", json={"question": "When was Acme Corp founded?"})
        data = response.json()
        assert "answer" in data

    def test_query_validation_empty_question(self, client: TestClient):
        response = client.post("/query", json={"question": ""})
        assert response.status_code == 422

    def test_query_validation_missing_question(self, client: TestClient):
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_query_validation_wrong_type(self, client: TestClient):
        response = client.post("/query", json={"question": 12345})
        assert response.status_code == 422


class TestDocumentsEndpoint:
    def test_documents_list_returns_200(self, client: TestClient):
        response = client.get("/documents")
        assert response.status_code == 200

    def test_documents_list_returns_array(self, client: TestClient):
        response = client.get("/documents")
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_documents_list_item_structure(self, client: TestClient):
        response = client.get("/documents")
        data = response.json()
        for item in data:
            assert "filename" in item
            assert "num_chunks" in item
