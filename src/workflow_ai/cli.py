"""Argparse CLI layer for running the full workflow pipeline.

Kept separate from scripts/run_pipeline.py so argument parsing and the
run/save orchestration are importable and testable without invoking a
subprocess.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from workflow_ai.config import DATA_RAW_DIR, MODEL_PATH
from workflow_ai.service import load_and_validate_workflow_data, run_pipeline, save_outputs

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the pipeline CLI."""
    parser = argparse.ArgumentParser(description="Run the full workflow risk pipeline end to end.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DATA_RAW_DIR / "workflows.csv"),
        help="Path to the workflow CSV file.",
    )
    parser.add_argument(
        "--train-if-missing",
        action="store_true",
        help="Train and persist a model first if none exists at MODEL_PATH, instead of failing.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: load, validate, run the pipeline, and save outputs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    df = load_and_validate_workflow_data(Path(args.data_path))
    result = run_pipeline(df, train_if_missing=args.train_if_missing, model_path=MODEL_PATH)
    save_outputs(result)

    logger.info(
        "Pipeline complete: %d predictions, %d alerts, %d recommendations",
        len(result.predictions),
        len(result.alerts),
        len(result.recommendations),
    )
