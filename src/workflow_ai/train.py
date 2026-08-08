"""Training orchestration: split, fit, evaluate, and persist the risk model.

train_and_evaluate performs no filesystem I/O -- it is a pure function
over an in-memory DataFrame, so it can be tested on small synthetic
frames without touching disk. save_model/save_metadata are the only
functions here that write to disk, and are meant to be called by
scripts/train_model.py.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from workflow_ai.config import MODEL_METADATA_PATH, MODEL_PATH, RANDOM_SEED, RISK_LABELS, TARGET_COLUMN
from workflow_ai.evaluate import EvaluationResult, evaluate_pipeline, format_evaluation_summary
from workflow_ai.feature_engineering import FEATURE_COLUMNS, prepare_feature_frame
from workflow_ai.models import BASELINE_BUILDERS, PRIMARY_MODEL_NAME, build_random_forest_pipeline

import joblib

logger = logging.getLogger(__name__)

DEFAULT_TEST_SIZE: float = 0.2


@dataclass(frozen=True)
class TrainingResult:
    """The fitted primary pipeline plus evaluation metrics and metadata."""

    pipeline: Pipeline
    primary_metrics: EvaluationResult
    baseline_metrics: dict[str, EvaluationResult]
    metadata: dict


def split_features_and_target(
    df: pd.DataFrame, test_size: float = DEFAULT_TEST_SIZE, random_state: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split of the raw workflow DataFrame on risk_label.

    Splitting the raw rows (before feature preparation) keeps
    workflow_id available on the resulting splits for any later
    reporting need, and keeps prepare_feature_frame's job purely
    feature selection.
    """
    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COLUMN],
    )


def train_and_evaluate(
    df: pd.DataFrame, test_size: float = DEFAULT_TEST_SIZE, random_state: int = RANDOM_SEED
) -> TrainingResult:
    """Split, fit the primary model and baselines, and evaluate all of them.

    prepare_feature_frame is called exactly once per split here -- the
    same shared path predict.py uses -- so training and inference never
    derive features differently.
    """
    train_df, test_df = split_features_and_target(df, test_size=test_size, random_state=random_state)

    X_train = prepare_feature_frame(train_df)
    y_train = train_df[TARGET_COLUMN]
    X_test = prepare_feature_frame(test_df)
    y_test = test_df[TARGET_COLUMN]

    primary_pipeline = build_random_forest_pipeline()
    primary_pipeline.fit(X_train, y_train)
    primary_metrics = evaluate_pipeline(primary_pipeline, X_test, y_test)
    logger.info("%s: %s", PRIMARY_MODEL_NAME, format_evaluation_summary(primary_metrics))

    baseline_metrics: dict[str, EvaluationResult] = {}
    for name, builder in BASELINE_BUILDERS.items():
        baseline_pipeline = builder()
        baseline_pipeline.fit(X_train, y_train)
        metrics = evaluate_pipeline(baseline_pipeline, X_test, y_test)
        baseline_metrics[name] = metrics
        logger.info("%s (baseline): %s", name, format_evaluation_summary(metrics))

    metadata = build_metadata(
        primary_metrics=primary_metrics,
        baseline_metrics=baseline_metrics,
        random_state=random_state,
        n_train=len(train_df),
        n_test=len(test_df),
    )

    return TrainingResult(
        pipeline=primary_pipeline,
        primary_metrics=primary_metrics,
        baseline_metrics=baseline_metrics,
        metadata=metadata,
    )


def build_metadata(
    primary_metrics: EvaluationResult,
    baseline_metrics: dict[str, EvaluationResult],
    random_state: int,
    n_train: int,
    n_test: int,
) -> dict:
    """Assemble the JSON-serializable metadata persisted alongside the model."""
    return {
        "model_type": "RandomForestClassifier",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "features": list(FEATURE_COLUMNS),
        "target": TARGET_COLUMN,
        "random_state": random_state,
        "n_train_rows": n_train,
        "n_test_rows": n_test,
        "risk_labels": list(RISK_LABELS),
        "test_metrics": primary_metrics.to_dict(),
        "baseline_comparison": {name: metrics.to_dict() for name, metrics in baseline_metrics.items()},
        "data_note": (
            "Trained on synthetic data (scripts/generate_demo_data.py); "
            "metrics are not representative of real-world performance."
        ),
    }


def save_model(pipeline: Pipeline, path: Path = MODEL_PATH) -> None:
    """Persist the complete fitted pipeline (preprocessing + classifier) as one artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("Saved trained model to %s", path)


def save_metadata(metadata: dict, path: Path = MODEL_METADATA_PATH) -> None:
    """Persist training metadata as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    logger.info("Saved model metadata to %s", path)
