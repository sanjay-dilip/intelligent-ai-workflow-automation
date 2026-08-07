"""Smoke tests for centralized configuration."""

from workflow_ai.config import DEFAULT_RULE_CONFIG, REQUIRED_COLUMNS, RISK_LABELS, RuleConfig


def test_rule_config_defaults_are_frozen() -> None:
    """RuleConfig should be an immutable dataclass with the documented defaults."""
    assert isinstance(DEFAULT_RULE_CONFIG, RuleConfig)
    assert DEFAULT_RULE_CONFIG.sla_warning_threshold == 0.80
    assert DEFAULT_RULE_CONFIG.high_error_threshold == 5


def test_required_columns_include_workflow_id() -> None:
    """The schema contract must always include the primary identifier."""
    assert "workflow_id" in REQUIRED_COLUMNS


def test_risk_labels_are_ordered_low_to_high() -> None:
    """Risk labels should be defined once, ordered low to high."""
    assert RISK_LABELS == ("low", "medium", "high")
