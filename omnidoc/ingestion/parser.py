"""
DocumentParser — multi-format file parser.

Supported formats: PDF, DOCX, XLSX, Markdown, HTML, PPTX, scanned images (OCR).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llama_index.core import Document

from omnidoc.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedDocument:
    """Intermediate representation after format-specific parsing."""

    source_path: str
    format: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    page_count: int = 0


class DocumentParser:
    """
    Dispatches raw files to format-specific sub-parsers and returns
    a normalised ``ParsedDocument`` ready for the chunking stage.

    Supported formats
    -----------------
    * PDF       — pdfplumber (text) + pytesseract (scanned OCR fallback)
    * DOCX      — python-docx
    * XLSX/CSV  — pandas + openpyxl (table→markdown conversion)
    * Markdown  — direct text read with frontmatter extraction
    * HTML      — BeautifulSoup4 boilerplate removal
    * PPTX      — python-pptx slide-by-slide extraction
    """

    SUPPORTED_FORMATS = {".pdf", ".docx", ".xlsx", ".csv", ".md", ".html", ".htm", ".pptx"}

    def parse(self, file_path: str | Path) -> ParsedDocument:
        """
        Parse *file_path* and return a :class:`ParsedDocument`.

        Parameters
        ----------
        file_path:
            Absolute or relative path to the source document.

        Raises
        ------
        ValueError
            When the file extension is not in :attr:`SUPPORTED_FORMATS`.
        FileNotFoundError
            When *file_path* does not exist on disk.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format '{suffix}'. Supported: {self.SUPPORTED_FORMATS}")

        logger.info("Parsing %s (format=%s)", path.name, suffix)

        dispatch = {
            ".pdf": self._parse_pdf,
            ".docx": self._parse_docx,
            ".xlsx": self._parse_xlsx,
            ".csv": self._parse_csv,
            ".md": self._parse_markdown,
            ".html": self._parse_html,
            ".htm": self._parse_html,
            ".pptx": self._parse_pptx,
        }
        return dispatch[suffix](path)

    def to_llama_document(self, parsed: ParsedDocument) -> Document:
        """Convert a :class:`ParsedDocument` to a LlamaIndex :class:`Document`."""
        return Document(
            text=parsed.text,
            metadata={
                "source": parsed.source_path,
                "format": parsed.format,
                "page_count": parsed.page_count,
                **parsed.metadata,
            },
        )

    # ------------------------------------------------------------------
    # Private per-format parsers
    # ------------------------------------------------------------------

    def _parse_pdf(self, path: Path) -> ParsedDocument:
        try:
            import pdfplumber  # type: ignore

            pages: list[str] = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    if not text.strip():
                        # OCR fallback for scanned pages
                        text = self._ocr_page(page)
                    pages.append(text)

            return ParsedDocument(
                source_path=str(path),
                format="pdf",
                text="\n\n".join(pages),
                page_count=len(pages),
                metadata={"file_name": path.name},
            )
        except ImportError:
            raise ImportError("Install pdfplumber: pip install pdfplumber")

    def _parse_docx(self, path: Path) -> ParsedDocument:
        try:
            from docx import Document as DocxDoc  # type: ignore

            doc = DocxDoc(str(path))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return ParsedDocument(
                source_path=str(path),
                format="docx",
                text="\n\n".join(paragraphs),
                page_count=1,
                metadata={"file_name": path.name},
            )
        except ImportError:
            raise ImportError("Install python-docx: pip install python-docx")

    def _parse_xlsx(self, path: Path) -> ParsedDocument:
        try:
            import pandas as pd  # type: ignore

            xl = pd.ExcelFile(str(path))
            sheets_md: list[str] = []
            for sheet in xl.sheet_names:
                df = xl.parse(sheet)
                sheets_md.append(f"## Sheet: {sheet}\n\n{df.to_markdown(index=False)}")

            return ParsedDocument(
                source_path=str(path),
                format="xlsx",
                text="\n\n".join(sheets_md),
                page_count=len(xl.sheet_names),
                metadata={"file_name": path.name, "sheets": xl.sheet_names},
            )
        except ImportError:
            raise ImportError("Install pandas + tabulate: pip install pandas tabulate openpyxl")

    def _parse_csv(self, path: Path) -> ParsedDocument:
        try:
            import pandas as pd  # type: ignore

            df = pd.read_csv(str(path))
            return ParsedDocument(
                source_path=str(path),
                format="csv",
                text=df.to_markdown(index=False),
                page_count=1,
                metadata={"file_name": path.name, "rows": len(df), "columns": list(df.columns)},
            )
        except ImportError:
            raise ImportError("Install pandas + tabulate: pip install pandas tabulate")

    def _parse_markdown(self, path: Path) -> ParsedDocument:
        import re

        raw = path.read_text(encoding="utf-8")
        # Strip YAML frontmatter if present
        frontmatter: dict[str, str] = {}
        fm_match = re.match(r"^---\n(.+?)\n---\n", raw, re.DOTALL)
        if fm_match:
            for line in fm_match.group(1).splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    frontmatter[k.strip()] = v.strip()
            raw = raw[fm_match.end():]

        return ParsedDocument(
            source_path=str(path),
            format="markdown",
            text=raw,
            page_count=1,
            metadata={"file_name": path.name, **frontmatter},
        )

    def _parse_html(self, path: Path) -> ParsedDocument:
        try:
            from bs4 import BeautifulSoup  # type: ignore

            html = path.read_text(encoding="utf-8")
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            return ParsedDocument(
                source_path=str(path),
                format="html",
                text=text,
                page_count=1,
                metadata={"file_name": path.name},
            )
        except ImportError:
            raise ImportError("Install beautifulsoup4: pip install beautifulsoup4")

    def _parse_pptx(self, path: Path) -> ParsedDocument:
        try:
            from pptx import Presentation  # type: ignore

            prs = Presentation(str(path))
            slides: list[str] = []
            for i, slide in enumerate(prs.slides, 1):
                texts = [shape.text for shape in slide.shapes if shape.has_text_frame]
                slides.append(f"### Slide {i}\n\n" + "\n".join(texts))

            return ParsedDocument(
                source_path=str(path),
                format="pptx",
                text="\n\n".join(slides),
                page_count=len(slides),
                metadata={"file_name": path.name},
            )
        except ImportError:
            raise ImportError("Install python-pptx: pip install python-pptx")

    @staticmethod
    def _ocr_page(page: Any) -> str:
        """Fallback OCR using pytesseract on a rendered page image."""
        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore

            img = page.to_image(resolution=300).original
            return pytesseract.image_to_string(img)
        except ImportError:
            logger.warning("pytesseract / Pillow not installed; OCR fallback skipped.")
            return ""
