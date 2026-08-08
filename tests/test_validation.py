"""Tests for the workflow data validation layer."""

from __future__ import annotations

import pandas as pd
import pytest

from workflow_ai.validation import validate_workflow_data

VALID_ROW = {
    "workflow_id": "WF-1000",
    "workflow_name": "Invoice Processing",
    "department": "Finance",
    "status": "open",
    "priority": "medium",
    "sla_hours": 48.0,
    "elapsed_hours": 20.0,
    "manual_steps": 4,
    "automated_steps": 6,
    "automation_score": 0.6,
    "error_count": 1,
    "rework_count": 0,
    "transaction_volume": 100,
    "avg_handling_time_minutes": 30.0,
    "pending_items": 5,
    "completion_rate": 0.9,
    "created_at": "2026-01-01",
    "updated_at": "2026-01-02",
}


def _make_df(overrides: dict | None = None, drop_columns: list[str] | None = None) -> pd.DataFrame:
    row = dict(VALID_ROW)
    if overrides:
        row.update(overrides)
    df = pd.DataFrame([row])
    if drop_columns:
        df = df.drop(columns=drop_columns)
    return df


def test_valid_dataset_has_no_errors() -> None:
    result = validate_workflow_data(_make_df())
    assert result.is_valid
    assert result.errors == []


def test_missing_required_columns() -> None:
    result = validate_workflow_data(_make_df(drop_columns=["sla_hours"]))
    assert not result.is_valid
    assert any(issue.check == "required_columns" for issue in result.errors)


def test_duplicate_workflow_id() -> None:
    df = pd.concat([_make_df(), _make_df()], ignore_index=True)
    result = validate_workflow_data(df)
    assert not result.is_valid
    assert any(issue.check == "duplicate_workflow_id" for issue in result.errors)


@pytest.mark.parametrize(
    "column",
    [
        "sla_hours",
        "elapsed_hours",
        "error_count",
        "rework_count",
        "manual_steps",
        "transaction_volume",
    ],
)
def test_negative_values_are_rejected(column: str) -> None:
    result = validate_workflow_data(_make_df({column: -1}))
    assert not result.is_valid
    checks = {issue.check for issue in result.errors}
    assert "negative_value" in checks or "impossible_sla" in checks


def test_negative_sla_hours_is_flagged_as_impossible_sla() -> None:
    result = validate_workflow_data(_make_df({"sla_hours": -10}))
    assert not result.is_valid
    checks = {issue.check for issue in result.errors}
    assert "impossible_sla" in checks
    assert "negative_value" in checks


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_completion_rate_out_of_bounds(value: float) -> None:
    result = validate_workflow_data(_make_df({"completion_rate": value}))
    assert not result.is_valid
    assert any(issue.check == "unit_interval_bounds" for issue in result.errors)


@pytest.mark.parametrize("value", [-0.1, 1.2])
def test_automation_score_out_of_bounds(value: float) -> None:
    result = validate_workflow_data(_make_df({"automation_score": value}))
    assert not result.is_valid
    assert any(issue.check == "unit_interval_bounds" for issue in result.errors)


def test_percentage_scale_hint_for_completion_rate_over_one() -> None:
    result = validate_workflow_data(_make_df({"completion_rate": 90.0}))
    message = next(
        issue.message for issue in result.errors if issue.check == "unit_interval_bounds"
    )
    assert "0-100" in message


def test_inconsistent_timestamps() -> None:
    result = validate_workflow_data(
        _make_df({"created_at": "2026-02-01", "updated_at": "2026-01-01"})
    )
    assert not result.is_valid
    assert any(issue.check == "inconsistent_timestamps" for issue in result.errors)


def test_missing_critical_value_is_null_error() -> None:
    result = validate_workflow_data(_make_df({"workflow_id": None}))
    assert not result.is_valid
    assert any(issue.check == "null_critical_field" for issue in result.errors)


def test_unknown_status_is_a_warning_not_an_error() -> None:
    result = validate_workflow_data(_make_df({"status": "mystery_status"}))
    assert result.is_valid
    assert any(issue.check == "unknown_status" for issue in result.warnings)


def test_zero_sla_hours_is_impossible() -> None:
    result = validate_workflow_data(_make_df({"sla_hours": 0}))
    assert not result.is_valid
    assert any(issue.check == "impossible_sla" for issue in result.errors)


def test_summary_reports_no_issues_for_valid_data() -> None:
    result = validate_workflow_data(_make_df())
    assert result.summary() == "No validation issues found."


def test_summary_lists_each_issue() -> None:
    result = validate_workflow_data(_make_df({"completion_rate": -1, "automation_score": -1}))
    lines = result.summary().splitlines()
    assert len(lines) == len(result.issues)
    assert len(result.issues) >= 2
