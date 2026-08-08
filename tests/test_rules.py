"""Tests for the deterministic rules engine, including threshold boundaries."""

from __future__ import annotations

import pandas as pd

from workflow_ai.config import DEFAULT_RULE_CONFIG, RuleConfig
from workflow_ai.feature_engineering import engineer_features
from workflow_ai.rules_engine import (
    APPROACHING_SLA,
    BACKLOG_PRESSURE,
    HIGH_ERROR,
    HIGH_REWORK,
    HIGH_RISK_ESCALATION,
    MANUAL_AUTOMATION_OPPORTUNITY,
    SLA_BREACH,
    evaluate_risk_escalation,
    evaluate_workflow_rules,
    risk_factors_from_rules,
)

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


def _engineered_row(overrides: dict | None = None) -> pd.Series:
    row = dict(BASE_ROW)
    if overrides:
        row.update(overrides)
    return engineer_features(pd.DataFrame([row])).iloc[0]


def _rule_types(row: pd.Series, rule_config: RuleConfig = DEFAULT_RULE_CONFIG) -> set[str]:
    return {trigger.rule_type for trigger in evaluate_workflow_rules(row, rule_config)}


def test_no_rules_trigger_on_safe_row() -> None:
    assert _rule_types(_engineered_row()) == set()


def test_sla_breach_triggers_at_full_utilization() -> None:
    row = _engineered_row({"elapsed_hours": 100.0})  # utilization == 1.0
    assert SLA_BREACH in _rule_types(row)
    assert APPROACHING_SLA not in _rule_types(row)


def test_sla_breach_triggers_above_full_utilization() -> None:
    row = _engineered_row({"elapsed_hours": 120.0})  # utilization == 1.2
    assert SLA_BREACH in _rule_types(row)


def test_approaching_sla_triggers_at_warning_threshold_boundary() -> None:
    row = _engineered_row(
        {"elapsed_hours": 80.0}
    )  # utilization == 0.80 == default warning threshold
    assert APPROACHING_SLA in _rule_types(row)
    assert SLA_BREACH not in _rule_types(row)


def test_approaching_sla_does_not_trigger_just_below_threshold() -> None:
    row = _engineered_row({"elapsed_hours": 79.0})  # utilization == 0.79
    assert APPROACHING_SLA not in _rule_types(row)


def test_high_error_triggers_at_threshold_boundary() -> None:
    row = _engineered_row({"error_count": DEFAULT_RULE_CONFIG.high_error_threshold})
    assert HIGH_ERROR in _rule_types(row)


def test_high_error_does_not_trigger_below_threshold() -> None:
    row = _engineered_row({"error_count": DEFAULT_RULE_CONFIG.high_error_threshold - 1})
    assert HIGH_ERROR not in _rule_types(row)


def test_high_rework_triggers_at_threshold_boundary() -> None:
    # rework_rate = rework_count / transaction_volume; use volume=100 for exact 0.10
    row = _engineered_row({"transaction_volume": 100, "rework_count": 10})
    assert HIGH_REWORK in _rule_types(row)


def test_high_rework_does_not_trigger_below_threshold() -> None:
    row = _engineered_row({"transaction_volume": 100, "rework_count": 9})
    assert HIGH_REWORK not in _rule_types(row)


def test_manual_automation_opportunity_triggers_on_high_manual_ratio() -> None:
    row = _engineered_row({"manual_steps": 7, "automated_steps": 3})  # ratio == 0.70
    assert MANUAL_AUTOMATION_OPPORTUNITY in _rule_types(row)


def test_manual_automation_opportunity_triggers_on_low_automation_score() -> None:
    row = _engineered_row({"automation_score": 0.30})
    assert MANUAL_AUTOMATION_OPPORTUNITY in _rule_types(row)


def test_manual_automation_opportunity_does_not_trigger_when_both_safe() -> None:
    row = _engineered_row({"manual_steps": 2, "automated_steps": 8, "automation_score": 0.8})
    assert MANUAL_AUTOMATION_OPPORTUNITY not in _rule_types(row)


def test_backlog_pressure_triggers_at_threshold_boundary() -> None:
    row = _engineered_row({"pending_items": DEFAULT_RULE_CONFIG.backlog_threshold})
    assert BACKLOG_PRESSURE in _rule_types(row)


def test_backlog_pressure_does_not_trigger_below_threshold() -> None:
    row = _engineered_row({"pending_items": DEFAULT_RULE_CONFIG.backlog_threshold - 1})
    assert BACKLOG_PRESSURE not in _rule_types(row)


def test_evaluate_risk_escalation_triggers_only_on_high() -> None:
    assert evaluate_risk_escalation("high").rule_type == HIGH_RISK_ESCALATION
    assert evaluate_risk_escalation("medium") is None
    assert evaluate_risk_escalation("low") is None
    assert evaluate_risk_escalation(None) is None


def test_risk_factors_from_rules_flags_sla_breach() -> None:
    row = _engineered_row({"elapsed_hours": 95.0})
    factors = risk_factors_from_rules(row, DEFAULT_RULE_CONFIG)
    assert any("SLA" in factor for factor in factors)
    assert not any("Error count" in factor for factor in factors)


def test_risk_factors_from_rules_returns_placeholder_when_nothing_triggers() -> None:
    factors = risk_factors_from_rules(_engineered_row(), DEFAULT_RULE_CONFIG)
    assert factors == ["No individual risk thresholds exceeded"]


def test_risk_factors_from_rules_is_never_empty() -> None:
    for overrides in ({}, {"elapsed_hours": 95.0}, {"error_count": 10}, {"pending_items": 200}):
        factors = risk_factors_from_rules(_engineered_row(overrides), DEFAULT_RULE_CONFIG)
        assert len(factors) >= 1
