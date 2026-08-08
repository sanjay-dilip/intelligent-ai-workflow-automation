"""CLI entry point: train, evaluate, and persist the workflow risk model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from workflow_ai.config import DATA_RAW_DIR, RANDOM_SEED
from workflow_ai.data_loader import load_workflow_data
from workflow_ai.evaluate import format_evaluation_summary
from workflow_ai.train import DEFAULT_TEST_SIZE, save_metadata, save_model, train_and_evaluate
from workflow_ai.validation import validate_workflow_data

logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point: load, validate, train, evaluate, and persist the risk model."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train the workflow risk classification model.")
    parser.add_argument(
        "--data-path", type=str, default=str(DATA_RAW_DIR / "workflows.csv"), help="Path to the workflow CSV file."
    )
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE, help="Fraction of rows held out for testing.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    args = parser.parse_args()

    df = load_workflow_data(Path(args.data_path))

    validation_result = validate_workflow_data(df)
    for warning in validation_result.warnings:
        logger.warning("%s: %s", warning.check, warning.message)
    if not validation_result.is_valid:
        logger.error("Workflow data failed validation:\n%s", validation_result.summary())
        raise SystemExit(1)

    result = train_and_evaluate(df, test_size=args.test_size, random_state=args.seed)

    save_model(result.pipeline)
    save_metadata(result.metadata)

    logger.info("Primary model (random_forest): %s", format_evaluation_summary(result.primary_metrics))
    for name, metrics in result.baseline_metrics.items():
        logger.info("Baseline (%s): %s", name, format_evaluation_summary(metrics))


if __name__ == "__main__":
    main()
