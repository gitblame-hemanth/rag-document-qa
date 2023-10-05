"""Document ingestion module — loaders, chunkers, and pipeline."""

from src.ingestion.chunkers import (
    BaseChunker,
    Chunk,
    FixedChunker,
    RecursiveChunker,
    SemanticChunker,
    get_chunker,
)
from src.ingestion.loaders import (
    BaseLoader,
    Document,
    DOCXLoader,
    MarkdownLoader,
    PDFLoader,
    TextLoader,
    get_loader,
)
from src.ingestion.pipeline import IngestPipeline, IngestResult

__all__ = [
    "BaseChunker",
    "BaseLoader",
    "Chunk",
    "DOCXLoader",
    "Document",
    "FixedChunker",
    "IngestPipeline",
    "IngestResult",
    "MarkdownLoader",
    "PDFLoader",
    "RecursiveChunker",
    "SemanticChunker",
    "TextLoader",
    "get_chunker",
    "get_loader",
]
