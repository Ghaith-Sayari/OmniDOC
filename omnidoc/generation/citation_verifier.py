"""
CitationVerifier — post-generation guard that ensures every sentence
in the LLM answer links to a cited chunk present in the context.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


class CitationVerifier:
    """
    Validates that:

    1. All ``[chunk:<id>]`` citations in the answer reference IDs that
       were actually supplied as context (no fabricated references).
    2. Sentences without any citation are flagged as potentially
       ungrounded (soft warning, not hard rejection).

    Parameters
    ----------
    strict:
        When ``True``, raise :class:`ValueError` on fabricated citations.
        When ``False``, log a warning instead (default ``False``).
    """

    CITATION_PATTERN = re.compile(r"\[chunk:([^\]]+)\]")

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict

    def verify(
        self,
        answer: str,
        context_chunks: List[Dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Run citation verification and return a report dict.

        Returns
        -------
        dict with keys:
            * ``valid``           — bool
            * ``cited_ids``       — list of IDs found in the answer
            * ``fabricated_ids``  — IDs cited but not in context
            * ``uncited_sentences`` — sentences without any citation
        """
        valid_ids = {str(c.get("id", "")) for c in context_chunks}
        cited_ids = list(set(self.CITATION_PATTERN.findall(answer)))
        fabricated = [cid for cid in cited_ids if cid not in valid_ids]

        sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
        uncited = [
            s for s in sentences
            if s and not self.CITATION_PATTERN.search(s)
        ]

        valid = len(fabricated) == 0
        report = {
            "valid": valid,
            "cited_ids": cited_ids,
            "fabricated_ids": fabricated,
            "uncited_sentences": uncited,
        }

        if not valid and self.strict:
            raise ValueError(
                f"Fabricated citations detected: {fabricated}. "
                "Answer references chunk IDs not present in context."
            )

        return report
