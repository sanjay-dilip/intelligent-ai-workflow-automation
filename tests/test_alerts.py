"""Tests for structured alerts and category-tagged recommendations."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from workflow_ai.alert_engine import generate_alerts
from workflow_ai.feature_engineering import engineer_features
from workflow_ai.recommendations import (
    ESCALATE,
    INVESTIGATE_ERRORS,
    MONITOR,
    NO_ACTION,
    generate_recommendations,
    generate_recommendations_for_workflow,
)
from workflow_ai.rules_engine import BACKLOG_PRESSURE, HIGH_ERROR, HIGH_RISK_ESCALATION, SLA_BREACH

BASE_ROW = {
    "workflow_id": "WF-1000",
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
}


def _row(overrides: dict | None = None) -> dict:
    row = dict(BASE_ROW)
    if overrides:
        row.update(overrides)
    return row


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# --- alert_engine ---------------------------------------------------------


def test_generate_alerts_produces_no_alerts_for_a_safe_workflow() -> None:
    alerts = generate_alerts(_df([_row()]))
    assert alerts == []


def test_generate_alerts_has_expected_fields_and_valid_timestamp() -> None:
    alerts = generate_alerts(_df([_row({"pending_items": 200})]))
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.workflow_id == "WF-1000"
    assert alert.type == BACKLOG_PRESSURE
    assert alert.severity == "warning"
    assert alert.alert_id
    assert alert.recommended_action
    datetime.fromisoformat(alert.generated_at)


def test_generate_alerts_includes_escalation_when_predicted_high() -> None:
    df = _df([_row()])
    alerts = generate_alerts(df, predictions={"WF-1000": "high"})
    assert any(alert.type == HIGH_RISK_ESCALATION and alert.severity == "critical" for alert in alerts)


def test_generate_alerts_omits_escalation_without_prediction() -> None:
    df = _df([_row()])
    alerts = generate_alerts(df)
    assert not any(alert.type == HIGH_RISK_ESCALATION for alert in alerts)


def test_generate_alerts_dedups_within_a_run() -> None:
    df = _df([_row({"error_count": 10}), _row({"error_count": 10})])  # same workflow_id twice
    alerts = generate_alerts(df)
    error_alerts = [alert for alert in alerts if alert.type == HIGH_ERROR]
    assert len(error_alerts) == 1


def test_generate_alerts_multiple_rules_produce_multiple_alerts() -> None:
    df = _df([_row({"elapsed_hours": 100.0, "error_count": 10})])  # SLA breach + high error
    alerts = generate_alerts(df)
    types = {alert.type for alert in alerts}
    assert {SLA_BREACH, HIGH_ERROR}.issubset(types)


# --- recommendations -------------------------------------------------------


def test_recommendations_tie_to_observed_signal() -> None:
    row = engineer_features(_df([_row({"error_count": 10})])).iloc[0]
    recommendations = generate_recommendations_for_workflow("WF-1000", row, predicted_risk=None)
    assert any(rec.category == INVESTIGATE_ERRORS for rec in recommendations)
    for rec in recommendations:
        assert rec.signal


def test_recommendations_escalate_on_sla_breach() -> None:
    row = engineer_features(_df([_row({"elapsed_hours": 100.0})])).iloc[0]
    recommendations = generate_recommendations_for_workflow("WF-1000", row, predicted_risk=None)
    assert any(rec.category == ESCALATE for rec in recommendations)


def test_recommendations_monitor_when_no_triggers_but_medium_risk() -> None:
    row = engineer_features(_df([_row()])).iloc[0]
    recommendations = generate_recommendations_for_workflow("WF-1000", row, predicted_risk="medium")
    assert len(recommendations) == 1
    assert recommendations[0].category == MONITOR


def test_recommendations_no_action_when_no_triggers_and_no_elevated_risk() -> None:
    row = engineer_features(_df([_row()])).iloc[0]
    recommendations = generate_recommendations_for_workflow("WF-1000", row, predicted_risk="low")
    assert len(recommendations) == 1
    assert recommendations[0].category == NO_ACTION


def test_generate_recommendations_covers_every_row() -> None:
    df = _df([_row({"workflow_id": "WF-1"}), _row({"workflow_id": "WF-2", "error_count": 10})])
    recommendations = generate_recommendations(df)
    workflow_ids = {rec.workflow_id for rec in recommendations}
    assert workflow_ids == {"WF-1", "WF-2"}
