"""Unit tests for DocumentParser."""

import pytest
from pathlib import Path
from omnidoc.ingestion.parser import DocumentParser, ParsedDocument


class TestDocumentParser:
    def setup_method(self):
        self.parser = DocumentParser()

    def test_unsupported_format_raises(self, tmp_path):
        fake = tmp_path / "file.xyz"
        fake.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported format"):
            self.parser.parse(fake)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            self.parser.parse("/nonexistent/path/file.pdf")

    def test_parse_markdown(self, tmp_path):
        md = tmp_path / "doc.md"
        md.write_text("---\ntitle: Test\n---\n# Hello\n\nWorld", encoding="utf-8")
        result = self.parser.parse(md)
        assert isinstance(result, ParsedDocument)
        assert result.format == "markdown"
        assert "Hello" in result.text
        assert result.metadata.get("title") == "Test"

    def test_parse_csv(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25", encoding="utf-8")
        result = self.parser.parse(csv_file)
        assert result.format == "csv"
        assert "Alice" in result.text

    def test_parse_html(self, tmp_path):
        html = tmp_path / "page.html"
        html.write_text("<html><body><p>Hello World</p><script>js</script></body></html>", encoding="utf-8")
        result = self.parser.parse(html)
        assert result.format == "html"
        assert "Hello World" in result.text
        assert "js" not in result.text  # script tag stripped

    def test_to_llama_document(self, tmp_path):
        from llama_index.core import Document

        md = tmp_path / "doc.md"
        md.write_text("# Test\n\nContent here.", encoding="utf-8")
        parsed = self.parser.parse(md)
        doc = self.parser.to_llama_document(parsed)
        assert isinstance(doc, Document)
        assert doc.metadata["format"] == "markdown"
