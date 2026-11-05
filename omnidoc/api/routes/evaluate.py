"""
Evaluate API route.

POST /evaluate  — run Ragas triad metrics on a provided Q&A dataset.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from omnidoc.config import Settings
from omnidoc.evaluation import RagasEvaluator, EvaluationReporter

router = APIRouter()


class EvaluationRequest(BaseModel):
    samples: List[Dict[str, Any]]  # each: {question, answer, contexts, ground_truth?}
    export_json: bool = True


def get_settings() -> Settings:
    return Settings()


@router.post("")
async def run_evaluation(
    request: EvaluationRequest,
    settings: Settings = Depends(get_settings),
) -> dict:
    """Evaluate *samples* with Ragas and return metric scores."""
    thresholds = {
        "context_relevance": settings.eval_threshold_context_relevance,
        "groundedness": settings.eval_threshold_groundedness,
        "answer_relevance": settings.eval_threshold_answer_relevance,
    }
    evaluator = RagasEvaluator(thresholds=thresholds)
    reporter = EvaluationReporter(output_dir=settings.eval_output_dir)

    records = evaluator.evaluate(request.samples)
    reporter.print_summary(records)

    response = {
        "total": len(records),
        "passed": sum(1 for r in records if r.passed),
        "results": [
            {"question": r.question, "scores": r.scores, "passed": r.passed}
            for r in records
        ],
    }

    if request.export_json:
        path = reporter.to_json(records)
        response["report_path"] = str(path)

    return response
