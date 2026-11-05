"""
Text utilities — normalisation and truncation helpers.
"""

from __future__ import annotations

import re


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters to a single space."""
    return re.sub(r"\s+", " ", text).strip()


def truncate_text(text: str, max_chars: int = 500, suffix: str = "…") -> str:
    """Return *text* truncated to *max_chars*, appending *suffix* if cut."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + suffix
