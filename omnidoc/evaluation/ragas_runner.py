"""
RagasEvaluator — drives the Ragas triad evaluation loop.

Metrics evaluated:
  * Context Relevance   ≥ 0.85
  * Groundedness (Faithfulness) ≥ 0.95
  * Answer Relevance    ≥ 0.90
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from omnidoc.utils import get_logger

logger = get_logger(__name__)

THRESHOLDS = {
    "context_relevance": 0.85,
    "groundedness": 0.95,
    "answer_relevance": 0.90,
}


@dataclass
class EvaluationRecord:
    """Single Q&A evaluation result."""

    question: str
    answer: str
    contexts: List[str]
    ground_truth: str = ""
    scores: Dict[str, float] = field(default_factory=dict)
    passed: bool = False


class RagasEvaluator:
    """
    Wraps the ``ragas`` library to score batches of Q&A pairs.

    Parameters
    ----------
    llm_model:
        LangChain-compatible LLM used by Ragas for faithfulness checking.
    embed_model:
        LangChain-compatible embedding model used for context relevance.
    thresholds:
        Override default metric pass/fail thresholds.
    """

    def __init__(
        self,
        llm_model=None,
        embed_model=None,
        thresholds: Dict[str, float] | None = None,
    ) -> None:
        self.thresholds = {**THRESHOLDS, **(thresholds or {})}
        self._llm = llm_model
        self._embed = embed_model

    def evaluate(self, records: List[Dict[str, Any]]) -> List[EvaluationRecord]:
        """
        Evaluate a list of Q&A dicts and return scored records.

        Each input dict must have keys:
        ``question``, ``answer``, ``contexts`` (list[str]),
        and optionally ``ground_truth``.

        Returns
        -------
        list[EvaluationRecord]
        """
        try:
            from datasets import Dataset  # type: ignore
            from ragas import evaluate  # type: ignore
            from ragas.metrics import (  # type: ignore
                answer_relevancy,
                context_precision,
                faithfulness,
            )
        except ImportError:
            raise ImportError("Install ragas: pip install ragas datasets")

        dataset = Dataset.from_list(records)
        metrics = [faithfulness, answer_relevancy, context_precision]

        result = evaluate(dataset=dataset, metrics=metrics)
        df = result.to_pandas()

        eval_records = []
        for i, row in df.iterrows():
            scores = {
                "groundedness": float(row.get("faithfulness", 0.0)),
                "answer_relevance": float(row.get("answer_relevancy", 0.0)),
                "context_relevance": float(row.get("context_precision", 0.0)),
            }
            passed = all(scores[k] >= self.thresholds[k] for k in scores)
            eval_records.append(
                EvaluationRecord(
                    question=records[i]["question"],
                    answer=records[i]["answer"],
                    contexts=records[i]["contexts"],
                    ground_truth=records[i].get("ground_truth", ""),
                    scores=scores,
                    passed=passed,
                )
            )
            logger.info(
                "Eval [%d/%d] passed=%s | %s",
                i + 1,
                len(records),
                passed,
                scores,
            )

        return eval_records
