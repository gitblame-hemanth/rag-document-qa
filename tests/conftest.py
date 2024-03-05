"""Shared test fixtures for RAG Document QA."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from src.ingestion.loaders import Document

# ---------------------------------------------------------------------------
# Temporary config
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_config(tmp_path: Path) -> Path:
    """Write a minimal config YAML and return its path."""
    config = {
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimensions": 1536,
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 512,
        },
        "chunking": {
            "strategy": "recursive",
            "chunk_size": 200,
            "overlap": 50,
        },
        "vectorstore": {
            "provider": "chroma",
            "collection_name": "test_documents",
            "persist_directory": str(tmp_path / "chroma_data"),
        },
        "retrieval": {
            "top_k": 3,
            "search_type": "similarity",
            "mmr_lambda": 0.5,
            "reranker_enabled": False,
        },
        "api": {
            "host": "127.0.0.1",
            "port": 8000,
        },
        "logging": {
            "level": "DEBUG",
            "format": "console",
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config, default_flow_style=False))
    return config_path


# ---------------------------------------------------------------------------
# Sample documents and chunks
# ---------------------------------------------------------------------------

SAMPLE_TEXT_CONTENT = (
    "Acme Corp was founded in 2010. The company specializes in cloud infrastructure "
    "and developer tools. Employees receive 15 days of paid vacation per year. "
    "The remote work policy allows up to three days per week from home."
)

SAMPLE_MD_CONTENT = """\
# Company Handbook

## Remote Work Policy

Employees may work remotely up to three days per week with manager approval.

## Vacation Policy

New employees receive 15 days of paid vacation during their first two years.
"""


@pytest.fixture()
def sample_documents() -> list[Document]:
    """Return a list of sample Document objects."""
    return [
        Document(
            content=(
                "Acme Corp was founded in 2010. The company specializes in cloud infrastructure."
            ),
            metadata={"source": "/docs/about.txt", "filename": "about.txt", "file_type": "txt"},
        ),
        Document(
            content="Employees receive 15 days of paid vacation per year.",
            metadata={"source": "/docs/hr.txt", "filename": "hr.txt", "file_type": "txt"},
        ),
        Document(
            content="The remote work policy allows up to three days per week from home.",
            metadata={"source": "/docs/hr.txt", "filename": "hr.txt", "file_type": "txt"},
        ),
    ]


@dataclass
class MockChunk:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    source_document_id: str = ""


@pytest.fixture()
def sample_chunks() -> list[MockChunk]:
    """Return a list of mock chunk objects."""
    return [
        MockChunk(
            content="Acme Corp was founded in 2010.",
            metadata={"source": "/docs/about.txt"},
            chunk_index=0,
        ),
        MockChunk(
            content="The company specializes in cloud infrastructure and developer tools.",
            metadata={"source": "/docs/about.txt"},
            chunk_index=1,
        ),
        MockChunk(
            content="Employees receive 15 days of paid vacation per year.",
            metadata={"source": "/docs/hr.txt"},
            chunk_index=0,
        ),
        MockChunk(
            content="The remote work policy allows up to three days per week from home.",
            metadata={"source": "/docs/hr.txt"},
            chunk_index=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Mock vectorstore
# ---------------------------------------------------------------------------


@dataclass
class MockSearchResult:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.9


@pytest.fixture()
def mock_vectorstore(sample_chunks):
    """Return a mock vectorstore that returns sample chunks on search."""
    store = MagicMock()
    store.search.return_value = [
        MockSearchResult(content=c.content, metadata=c.metadata, score=0.9 - i * 0.1)
        for i, c in enumerate(sample_chunks)
    ]
    store.add_documents.return_value = None
    store.collection_name = "test_documents"
    return store


# ---------------------------------------------------------------------------
# Mock LLM / chain
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_llm_response() -> str:
    return "Acme Corp was founded in 2010 and specializes in cloud infrastructure."


@pytest.fixture()
def mock_chain(mock_llm_response):
    """Return a mock RAG chain that returns a fixed response."""
    chain = MagicMock()
    chain.invoke.return_value = {"answer": mock_llm_response}
    return chain


# ---------------------------------------------------------------------------
# Temp files for loader tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def txt_file(tmp_path: Path) -> Path:
    """Create a temp .txt file."""
    p = tmp_path / "test_doc.txt"
    p.write_text(SAMPLE_TEXT_CONTENT, encoding="utf-8")
    return p


@pytest.fixture()
def md_file(tmp_path: Path) -> Path:
    """Create a temp .md file."""
    p = tmp_path / "test_doc.md"
    p.write_text(SAMPLE_MD_CONTENT, encoding="utf-8")
    return p


@pytest.fixture()
def empty_file(tmp_path: Path) -> Path:
    """Create an empty temp file."""
    p = tmp_path / "empty.txt"
    p.write_text("", encoding="utf-8")
    return p
