"""
EvaluationReporter — renders evaluation results to console, JSON, or CSV.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from omnidoc.evaluation.ragas_runner import EvaluationRecord
from omnidoc.utils import get_logger

logger = get_logger(__name__)


class EvaluationReporter:
    """
    Formats and exports evaluation results.

    Parameters
    ----------
    output_dir:
        Directory where JSON / CSV reports are written.
    """

    def __init__(self, output_dir: str | Path = "data/evaluation") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def print_summary(self, records: List[EvaluationRecord]) -> None:
        """Print a human-readable summary table to stdout."""
        total = len(records)
        passed = sum(1 for r in records if r.passed)
        avg_scores: dict[str, float] = {}

        for key in ["groundedness", "answer_relevance", "context_relevance"]:
            vals = [r.scores.get(key, 0.0) for r in records]
            avg_scores[key] = sum(vals) / len(vals) if vals else 0.0

        print("\n" + "=" * 60)
        print("  OmniDoc RAG — Evaluation Summary")
        print("=" * 60)
        print(f"  Total samples : {total}")
        print(f"  Passed        : {passed} / {total}  ({100 * passed / max(total, 1):.1f}%)")
        print("-" * 60)
        for metric, score in avg_scores.items():
            status = "✓" if score >= 0.85 else "✗"
            print(f"  {status}  {metric:<24} avg = {score:.4f}")
        print("=" * 60 + "\n")

    def to_json(self, records: List[EvaluationRecord], filename: str = "eval_results.json") -> Path:
        """Write full results to a JSON file and return its path."""
        out = self.output_dir / filename
        data = [
            {
                "question": r.question,
                "answer": r.answer,
                "scores": r.scores,
                "passed": r.passed,
            }
            for r in records
        ]
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Evaluation JSON written to %s", out)
        return out

    def to_csv(self, records: List[EvaluationRecord], filename: str = "eval_results.csv") -> Path:
        """Write a flat CSV summary and return its path."""
        import csv

        out = self.output_dir / filename
        fieldnames = ["question", "passed", "groundedness", "answer_relevance", "context_relevance"]
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(
                    {
                        "question": r.question,
                        "passed": r.passed,
                        **r.scores,
                    }
                )
        logger.info("Evaluation CSV written to %s", out)
        return out
