"""
Ragas evaluation runner script.

Usage:
    python scripts/run_evaluation.py data/evaluation/samples.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from omnidoc.config import Settings
from omnidoc.evaluation import RagasEvaluator, EvaluationReporter
from omnidoc.utils import get_logger

logger = get_logger("run_evaluation")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_evaluation.py <samples.json>")
        sys.exit(1)

    samples_path = Path(sys.argv[1])
    if not samples_path.exists():
        logger.error("File not found: %s", samples_path)
        sys.exit(1)

    with open(samples_path, encoding="utf-8") as f:
        samples = json.load(f)

    settings = Settings()
    thresholds = {
        "context_relevance": settings.eval_threshold_context_relevance,
        "groundedness": settings.eval_threshold_groundedness,
        "answer_relevance": settings.eval_threshold_answer_relevance,
    }

    evaluator = RagasEvaluator(thresholds=thresholds)
    reporter = EvaluationReporter(output_dir=settings.eval_output_dir)

    logger.info("Running Ragas evaluation on %d samples…", len(samples))
    records = evaluator.evaluate(samples)
    reporter.print_summary(records)
    reporter.to_json(records)
    reporter.to_csv(records)
    logger.info("Evaluation complete. Results in %s", settings.eval_output_dir)


if __name__ == "__main__":
    main()
