"""Tests for the shared preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from workflow_ai.feature_engineering import prepare_feature_frame
from workflow_ai.preprocessing import build_preprocessing_pipeline

RAW_ROWS = pd.DataFrame(
    [
        {
            "sla_hours": 100.0,
            "elapsed_hours": 80.0,
            "manual_steps": 3,
            "automated_steps": 7,
            "automation_score": 0.7,
            "error_count": 2,
            "rework_count": 1,
            "transaction_volume": 50,
            "avg_handling_time_minutes": 20.0,
            "pending_items": 10,
            "completion_rate": 0.9,
            "department": "Finance",
            "status": "open",
            "priority": "medium",
        },
        {
            "sla_hours": 40.0,
            "elapsed_hours": 55.0,
            "manual_steps": 8,
            "automated_steps": 2,
            "automation_score": 0.2,
            "error_count": 6,
            "rework_count": 4,
            "transaction_volume": 30,
            "avg_handling_time_minutes": 90.0,
            "pending_items": 40,
            "completion_rate": 0.5,
            "department": "IT",
            "status": "in_progress",
            "priority": "high",
        },
    ]
)


def test_pipeline_fits_and_transforms_without_nans() -> None:
    features = prepare_feature_frame(RAW_ROWS)
    pipeline = build_preprocessing_pipeline()

    transformed = pipeline.fit_transform(features)

    assert transformed.shape[0] == len(features)
    assert not np.isnan(transformed).any()


def test_pipeline_handles_unseen_categorical_value_at_inference() -> None:
    features = prepare_feature_frame(RAW_ROWS)
    pipeline = build_preprocessing_pipeline()
    pipeline.fit(features)

    unseen = features.copy()
    unseen.loc[0, "department"] = "Unseen Department"

    transformed = pipeline.transform(unseen)

    assert transformed.shape[0] == len(unseen)
    assert not np.isnan(transformed).any()


def test_pipeline_imputes_missing_numeric_value() -> None:
    features = prepare_feature_frame(RAW_ROWS)
    features.loc[0, "automation_score"] = np.nan
    pipeline = build_preprocessing_pipeline()

    transformed = pipeline.fit_transform(features)

    assert not np.isnan(transformed).any()
