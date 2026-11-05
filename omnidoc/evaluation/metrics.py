"""
Metric dataclasses — typed wrappers for the three Ragas triad scores.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContextRelevance:
    """
    Measures the fraction of retrieved context that is relevant to the
    question. High precision, low noise.

    Target: ≥ 0.85
    """

    score: float
    threshold: float = 0.85

    @property
    def passed(self) -> bool:
        return self.score >= self.threshold


@dataclass
class Groundedness:
    """
    Also called Faithfulness. Checks that every generated sentence is
    entailed by one or more retrieved context chunks.

    Target: ≥ 0.95
    """

    score: float
    threshold: float = 0.95

    @property
    def passed(self) -> bool:
        return self.score >= self.threshold


@dataclass
class AnswerRelevance:
    """
    Checks whether the final response directly addresses the user's
    explicit intent (penalises off-topic or incomplete answers).

    Target: ≥ 0.90
    """

    score: float
    threshold: float = 0.90

    @property
    def passed(self) -> bool:
        return self.score >= self.threshold
