"""Tests for document loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.loaders import (
    Document,
    MarkdownLoader,
    TextLoader,
    get_loader,
)


class TestTextLoader:
    def test_text_loader_returns_single_document(self, txt_file: Path):
        loader = TextLoader()
        docs = loader.load(txt_file)

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert "Acme Corp" in docs[0].content
        assert docs[0].metadata["file_type"] == "txt"
        assert docs[0].metadata["filename"] == "test_doc.txt"

    def test_text_loader_metadata_has_source(self, txt_file: Path):
        loader = TextLoader()
        docs = loader.load(txt_file)

        assert docs[0].metadata["source"] != ""
        assert str(txt_file.resolve()) in docs[0].metadata["source"]

    def test_text_loader_file_not_found(self, tmp_path: Path):
        loader = TextLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "nonexistent.txt")

    def test_text_loader_empty_file(self, empty_file: Path):
        loader = TextLoader()
        with pytest.raises(ValueError, match="empty"):
            loader.load(empty_file)


class TestMarkdownLoader:
    def test_markdown_loader_strips_frontmatter(self, tmp_path: Path):
        md_path = tmp_path / "with_front.md"
        md_path.write_text(
            "---\ntitle: Test\nauthor: Bot\n---\n\n# Hello\n\nThis is content.",
            encoding="utf-8",
        )
        loader = MarkdownLoader()
        docs = loader.load(md_path)

        assert len(docs) == 1
        assert "title:" not in docs[0].content
        assert "Hello" in docs[0].content
        assert docs[0].metadata["section"] == "Hello"

    def test_markdown_loader_basic(self, md_file: Path):
        loader = MarkdownLoader()
        docs = loader.load(md_file)

        assert len(docs) == 1
        assert "Remote Work Policy" in docs[0].content
        assert docs[0].metadata["file_type"] == "md"

    def test_markdown_loader_extracts_first_heading(self, md_file: Path):
        loader = MarkdownLoader()
        docs = loader.load(md_file)

        assert docs[0].metadata.get("section") == "Company Handbook"

    def test_markdown_loader_empty_after_frontmatter(self, tmp_path: Path):
        md_path = tmp_path / "only_front.md"
        md_path.write_text("---\ntitle: Nothing\n---\n", encoding="utf-8")
        loader = MarkdownLoader()
        with pytest.raises(ValueError, match="no content"):
            loader.load(md_path)


class TestGetLoaderFactory:
    def test_get_loader_txt(self):
        loader = get_loader("document.txt")
        assert isinstance(loader, TextLoader)

    def test_get_loader_md(self):
        loader = get_loader("readme.md")
        assert isinstance(loader, MarkdownLoader)

    def test_get_loader_markdown_extension(self):
        loader = get_loader("notes.markdown")
        assert isinstance(loader, MarkdownLoader)

    def test_get_loader_case_insensitive(self):
        loader = get_loader("FILE.TXT")
        assert isinstance(loader, TextLoader)

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported file extension"):
            get_loader("data.xlsx")

    def test_unsupported_format_message_lists_supported(self):
        with pytest.raises(ValueError, match=r"\.txt"):
            get_loader("archive.zip")
