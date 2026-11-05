"""
File utilities — format detection and document discovery.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".csv", ".md", ".html", ".htm", ".pptx"}


def resolve_format(file_path: str | Path) -> str:
    """Return the lowercase extension (without dot) for *file_path*."""
    return Path(file_path).suffix.lower().lstrip(".")


def list_documents(directory: str | Path, recursive: bool = True) -> Generator[Path, None, None]:
    """
    Yield all supported document paths under *directory*.

    Parameters
    ----------
    directory:
        Root folder to scan.
    recursive:
        When ``True`` (default), scan subdirectories as well.
    """
    root = Path(directory)
    glob = "**/*" if recursive else "*"
    for path in root.glob(glob):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path
