"""Tests for KPI aggregation formulas."""

from __future__ import annotations

import pandas as pd

from workflow_ai.config import RISK_LABELS
from workflow_ai.kpi_engine import (
    build_kpi_summary,
    compute_department_breakdown,
    compute_efficiency_kpis,
    compute_risk_distribution,
    compute_sla_kpis,
    compute_volume_kpis,
)

BASE_ROW = {
    "workflow_id": "WF-1000",
    "workflow_name": "Invoice Processing",
    "department": "Finance",
    "status": "open",
    "priority": "medium",
    "sla_hours": 100.0,
    "elapsed_hours": 40.0,
    "manual_steps": 2,
    "automated_steps": 8,
    "automation_score": 0.8,
    "error_count": 1,
    "rework_count": 1,
    "transaction_volume": 50,
    "avg_handling_time_minutes": 25.0,
    "pending_items": 5,
    "completion_rate": 0.9,
    "created_at": pd.Timestamp("2026-01-01"),
    "updated_at": pd.Timestamp("2026-01-02"),
    "risk_label": "low",
}


def _row(overrides: dict | None = None) -> dict:
    row = dict(BASE_ROW)
    if overrides:
        row.update(overrides)
    return row


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_compute_volume_kpis_counts_total_and_breakdowns() -> None:
    df = _df(
        [
            _row(
                {
                    "workflow_id": "WF-1",
                    "department": "Finance",
                    "status": "open",
                    "priority": "low",
                }
            ),
            _row(
                {
                    "workflow_id": "WF-2",
                    "department": "IT",
                    "status": "completed",
                    "priority": "high",
                }
            ),
        ]
    )
    result = compute_volume_kpis(df)
    assert result["total_workflows"] == 2
    assert result["by_department"] == {"Finance": 1, "IT": 1}
    assert result["by_status"] == {"open": 1, "completed": 1}
    assert result["by_priority"] == {"low": 1, "high": 1}


def test_compute_sla_kpis_breach_and_approaching_are_mutually_exclusive() -> None:
    df = _df(
        [
            _row(
                {"workflow_id": "WF-breach", "sla_hours": 100.0, "elapsed_hours": 100.0}
            ),  # 1.0 -> breach
            _row(
                {"workflow_id": "WF-approaching", "sla_hours": 100.0, "elapsed_hours": 85.0}
            ),  # 0.85 -> approaching
            _row(
                {"workflow_id": "WF-safe", "sla_hours": 100.0, "elapsed_hours": 20.0}
            ),  # 0.2 -> neither
        ]
    )
    result = compute_sla_kpis(df)
    assert result["sla_breach_count"] == 1
    assert result["sla_approaching_count"] == 1
    assert result["sla_breach_rate"] == 1 / 3
    assert result["avg_sla_utilization"] > 0


def test_compute_efficiency_kpis_returns_expected_keys() -> None:
    df = _df([_row(), _row({"workflow_id": "WF-2"})])
    result = compute_efficiency_kpis(df)
    expected_keys = {
        "avg_manual_step_ratio",
        "avg_automation_score",
        "avg_completion_rate",
        "avg_handling_time_minutes",
        "avg_error_rate",
        "avg_rework_rate",
    }
    assert set(result.keys()) == expected_keys


def test_compute_risk_distribution_covers_all_labels_including_zero_counts() -> None:
    risk_labels = pd.Series(["low", "low", "high"])
    result = compute_risk_distribution(risk_labels)
    assert result["counts"] == {"low": 2, "medium": 0, "high": 1}
    assert result["shares"]["medium"] == 0.0
    assert set(result["counts"].keys()) == set(RISK_LABELS)


def test_compute_department_breakdown_groups_by_department() -> None:
    df = _df(
        [
            _row({"workflow_id": "WF-1", "department": "Finance"}),
            _row({"workflow_id": "WF-2", "department": "Finance"}),
            _row({"workflow_id": "WF-3", "department": "IT"}),
        ]
    )
    result = compute_department_breakdown(df)
    by_department = {entry["department"]: entry["workflow_count"] for entry in result}
    assert by_department == {"Finance": 2, "IT": 1}


def test_build_kpi_summary_includes_risk_only_when_provided() -> None:
    df = _df([_row()])
    without_risk = build_kpi_summary(df)
    with_risk = build_kpi_summary(df, risk_labels=pd.Series(["low"]))

    assert "risk" not in without_risk
    assert "risk" in with_risk
    assert set(without_risk.keys()) == {"volume", "sla", "efficiency", "by_department"}
