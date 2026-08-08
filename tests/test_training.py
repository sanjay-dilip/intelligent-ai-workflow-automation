"""Tests for training orchestration: split, fit, evaluate, persist."""

from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from workflow_ai.config import LEAKAGE_COLUMNS, RANDOM_SEED, RISK_LABELS, TARGET_COLUMN
from workflow_ai.evaluate import evaluate_pipeline
from workflow_ai.feature_engineering import FEATURE_COLUMNS, prepare_feature_frame
from workflow_ai.models import BASELINE_BUILDERS, build_dummy_baseline
from workflow_ai.train import (
    build_metadata,
    save_metadata,
    save_model,
    split_features_and_target,
    train_and_evaluate,
)

_ROWS_PER_CLASS = 20


def _synthetic_dataset(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Small in-memory dataset with signal correlated to risk_label, for fast tests."""
    rng = np.random.default_rng(seed)
    rows = []
    risk_bias = {"low": 0.1, "medium": 0.5, "high": 0.9}
    for label in RISK_LABELS:
        bias = risk_bias[label]
        for i in range(_ROWS_PER_CLASS):
            sla_hours = 100.0
            elapsed_hours = sla_hours * np.clip(bias + rng.normal(0, 0.05), 0.01, 1.5)
            error_count = max(0, int(rng.normal(bias * 10, 1)))
            rework_count = max(0, int(rng.normal(bias * 6, 1)))
            manual_steps = int(np.clip(bias * 10 + rng.normal(0, 1), 0, 10))
            automated_steps = 10 - manual_steps
            rows.append(
                {
                    "workflow_id": f"WF-{label}-{i}",
                    "workflow_name": "Invoice Processing",
                    "department": "Finance",
                    "status": "open",
                    "priority": "medium",
                    "sla_hours": sla_hours,
                    "elapsed_hours": elapsed_hours,
                    "manual_steps": manual_steps,
                    "automated_steps": automated_steps,
                    "automation_score": float(np.clip(1 - bias + rng.normal(0, 0.05), 0, 1)),
                    "error_count": error_count,
                    "rework_count": rework_count,
                    "transaction_volume": 50,
                    "avg_handling_time_minutes": 25.0,
                    "pending_items": int(np.clip(bias * 150 + rng.normal(0, 5), 0, None)),
                    "completion_rate": float(np.clip(1 - bias + rng.normal(0, 0.05), 0, 1)),
                    "created_at": pd.Timestamp("2026-01-01"),
                    "updated_at": pd.Timestamp("2026-01-02"),
                    "risk_label": label,
                }
            )
    return pd.DataFrame(rows)


def test_split_features_and_target_is_stratified() -> None:
    df = _synthetic_dataset()
    train_df, test_df = split_features_and_target(df, test_size=0.3, random_state=RANDOM_SEED)
    assert set(train_df[TARGET_COLUMN]) == set(RISK_LABELS)
    assert set(test_df[TARGET_COLUMN]) == set(RISK_LABELS)


def test_train_and_evaluate_returns_fitted_primary_pipeline() -> None:
    df = _synthetic_dataset()
    result = train_and_evaluate(df)
    classifier = result.pipeline.named_steps["classifier"]
    assert set(classifier.classes_).issubset(set(RISK_LABELS))
    predictions = result.pipeline.predict(prepare_feature_frame(df))
    assert set(predictions).issubset(set(RISK_LABELS))


def test_train_and_evaluate_includes_all_baselines() -> None:
    df = _synthetic_dataset()
    result = train_and_evaluate(df)
    assert set(result.baseline_metrics.keys()) == set(BASELINE_BUILDERS.keys())


def test_train_and_evaluate_metadata_has_required_fields() -> None:
    df = _synthetic_dataset()
    result = train_and_evaluate(df)
    metadata = result.metadata

    assert metadata["model_type"] == "RandomForestClassifier"
    datetime.fromisoformat(metadata["trained_at"])
    assert metadata["features"] == list(FEATURE_COLUMNS)
    assert metadata["target"] == TARGET_COLUMN
    assert metadata["random_state"] == RANDOM_SEED
    assert "test_metrics" in metadata
    assert "baseline_comparison" in metadata


def test_build_metadata_matches_train_and_evaluate_output() -> None:
    df = _synthetic_dataset()
    result = train_and_evaluate(df)
    rebuilt = build_metadata(
        primary_metrics=result.primary_metrics,
        baseline_metrics=result.baseline_metrics,
        random_state=RANDOM_SEED,
        n_train=10,
        n_test=5,
    )
    assert rebuilt["n_train_rows"] == 10
    assert rebuilt["n_test_rows"] == 5


def test_save_model_and_reload_predicts_consistently(tmp_path) -> None:
    df = _synthetic_dataset()
    result = train_and_evaluate(df)
    model_path = tmp_path / "model.joblib"

    save_model(result.pipeline, path=model_path)
    reloaded = joblib.load(model_path)

    features = prepare_feature_frame(df)
    original_predictions = result.pipeline.predict(features)
    reloaded_predictions = reloaded.predict(features)
    assert list(original_predictions) == list(reloaded_predictions)


def test_save_metadata_writes_valid_json(tmp_path) -> None:
    df = _synthetic_dataset()
    result = train_and_evaluate(df)
    metadata_path = tmp_path / "metadata.json"

    save_metadata(result.metadata, path=metadata_path)
    with metadata_path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)

    assert loaded == result.metadata


def test_evaluate_pipeline_returns_expected_keys() -> None:
    df = _synthetic_dataset()
    train_df, test_df = split_features_and_target(df, test_size=0.3, random_state=RANDOM_SEED)
    pipeline = build_dummy_baseline()
    pipeline.fit(prepare_feature_frame(train_df), train_df[TARGET_COLUMN])

    result = evaluate_pipeline(pipeline, prepare_feature_frame(test_df), test_df[TARGET_COLUMN])

    assert result.labels == list(RISK_LABELS)
    assert len(result.confusion_matrix) == 3
    assert all(len(row) == 3 for row in result.confusion_matrix)


class _PerfectPredictor:
    """A minimal predict()-only stub that always echoes the true labels back."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["_true_label"].to_numpy()


def test_evaluate_pipeline_perfect_predictions_score_one() -> None:
    df = _synthetic_dataset()
    _, test_df = split_features_and_target(df, test_size=0.3, random_state=RANDOM_SEED)
    features = prepare_feature_frame(test_df).copy()
    features["_true_label"] = test_df[TARGET_COLUMN].to_numpy()

    result = evaluate_pipeline(_PerfectPredictor(), features, test_df[TARGET_COLUMN])

    assert result.accuracy == 1.0
    assert result.f1_macro == 1.0


def test_evaluate_pipeline_handles_missing_class_in_split() -> None:
    df = _synthetic_dataset()
    train_df, test_df = split_features_and_target(df, test_size=0.3, random_state=RANDOM_SEED)
    test_df = test_df[test_df[TARGET_COLUMN] != "high"]
    pipeline = build_dummy_baseline()
    pipeline.fit(prepare_feature_frame(train_df), train_df[TARGET_COLUMN])

    result = evaluate_pipeline(pipeline, prepare_feature_frame(test_df), test_df[TARGET_COLUMN])

    assert result.support["high"] == 0


def test_leakage_columns_never_enter_pipeline_input() -> None:
    df = _synthetic_dataset()

    features = prepare_feature_frame(df)
    assert not set(features.columns) & set(LEAKAGE_COLUMNS)

    result = train_and_evaluate(df)
    preprocessing_step = result.pipeline.named_steps["preprocessing"]
    fitted_input_names = set(preprocessing_step.feature_names_in_)
    assert not fitted_input_names & set(LEAKAGE_COLUMNS)
