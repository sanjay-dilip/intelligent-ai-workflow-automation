"""Tests for operational feature engineering."""

from __future__ import annotations

import pandas as pd

from workflow_ai.feature_engineering import (
    FEATURE_COLUMNS,
    compute_backlog_pressure,
    compute_error_rate,
    compute_manual_step_ratio,
    compute_rework_rate,
    compute_sla_utilization,
    engineer_features,
    prepare_feature_frame,
)

BASE_ROW = {
    "sla_hours": 100.0,
    "elapsed_hours": 80.0,
    "manual_steps": 3,
    "automated_steps": 7,
    "automation_score": 0.6,
    "error_count": 2,
    "rework_count": 1,
    "transaction_volume": 50,
    "avg_handling_time_minutes": 25.0,
    "pending_items": 10,
    "completion_rate": 0.85,
    "department": "Finance",
    "status": "open",
    "priority": "medium",
}


def _df(overrides: dict | None = None) -> pd.DataFrame:
    row = dict(BASE_ROW)
    if overrides:
        row.update(overrides)
    return pd.DataFrame([row])


def test_sla_utilization() -> None:
    result = compute_sla_utilization(_df())
    assert result.iloc[0] == 0.8


def test_manual_step_ratio() -> None:
    result = compute_manual_step_ratio(_df())
    assert result.iloc[0] == 0.3


def test_error_rate() -> None:
    result = compute_error_rate(_df())
    assert result.iloc[0] == 0.04


def test_rework_rate() -> None:
    result = compute_rework_rate(_df())
    assert result.iloc[0] == 0.02


def test_backlog_pressure() -> None:
    result = compute_backlog_pressure(_df())
    assert result.iloc[0] == 0.2


def test_sla_utilization_division_by_zero_returns_zero() -> None:
    result = compute_sla_utilization(_df({"sla_hours": 0}))
    assert result.iloc[0] == 0.0


def test_error_rate_division_by_zero_returns_zero() -> None:
    result = compute_error_rate(_df({"transaction_volume": 0}))
    assert result.iloc[0] == 0.0


def test_manual_step_ratio_zero_total_steps_returns_zero() -> None:
    result = compute_manual_step_ratio(_df({"manual_steps": 0, "automated_steps": 0}))
    assert result.iloc[0] == 0.0


def test_missing_denominator_returns_zero_not_nan() -> None:
    df = _df()
    df.loc[0, "sla_hours"] = None
    result = compute_sla_utilization(df)
    assert result.iloc[0] == 0.0
    assert not result.isna().any()


def test_engineer_features_adds_all_expected_columns() -> None:
    result = engineer_features(_df())
    for column in ("sla_utilization", "manual_step_ratio", "error_rate", "rework_rate", "backlog_pressure"):
        assert column in result.columns


def test_engineer_features_is_deterministic() -> None:
    df = _df()
    first = engineer_features(df)
    second = engineer_features(df)
    pd.testing.assert_frame_equal(first, second)


def test_engineer_features_does_not_mutate_input() -> None:
    df = _df()
    original_columns = set(df.columns)
    engineer_features(df)
    assert set(df.columns) == original_columns


def test_prepare_feature_frame_selects_declared_columns_only() -> None:
    result = prepare_feature_frame(_df())
    assert list(result.columns) == list(FEATURE_COLUMNS)


def test_prepare_feature_frame_excludes_leakage_columns() -> None:
    df = _df({"workflow_id": "WF-1", "risk_label": "high"})
    result = prepare_feature_frame(df)
    assert "workflow_id" not in result.columns
    assert "risk_label" not in result.columns
