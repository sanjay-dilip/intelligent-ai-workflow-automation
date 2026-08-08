"""Structured alerts built from rules_engine triggers.

One run of generate_alerts produces zero or more Alert records per
workflow, one per distinct rule that fired. Alerts are deduplicated
within a run so the same (workflow_id, rule_type) pair never appears
twice, even if generate_alerts is called with overlapping rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from workflow_ai.config import DEFAULT_RULE_CONFIG, RuleConfig, WORKFLOW_ID_COLUMN
from workflow_ai.feature_engineering import engineer_features
from workflow_ai.rules_engine import RuleTrigger, evaluate_risk_escalation, evaluate_workflow_rules


@dataclass(frozen=True)
class Alert:
    """A structured alert for one workflow and one triggered rule."""

    alert_id: str
    workflow_id: str
    severity: str
    type: str
    message: str
    recommended_action: str
    generated_at: str


def _alert_from_trigger(workflow_id: str, trigger: RuleTrigger, generated_at: str) -> Alert:
    return Alert(
        alert_id=f"ALT-{workflow_id}-{trigger.rule_type}",
        workflow_id=workflow_id,
        severity=trigger.severity,
        type=trigger.rule_type,
        message=trigger.message,
        recommended_action=trigger.recommended_action,
        generated_at=generated_at,
    )


def generate_alerts(
    df: pd.DataFrame,
    predictions: dict[str, str] | None = None,
    rule_config: RuleConfig = DEFAULT_RULE_CONFIG,
) -> list[Alert]:
    """Generate deduplicated alerts for every rule that fires across df.

    Args:
        df: Raw workflow rows.
        predictions: Optional mapping of workflow_id -> predicted_risk,
            used to evaluate the high-risk-escalation rule. Workflows
            missing from this mapping are simply not evaluated for
            escalation.
        rule_config: Thresholds passed through to the rules engine.

    Returns:
        One Alert per distinct (workflow_id, rule_type) that fired,
        in row order.
    """
    predictions = predictions or {}
    generated_at = datetime.now(timezone.utc).isoformat()
    engineered = engineer_features(df)

    alerts: list[Alert] = []
    seen: set[tuple[str, str]] = set()

    for _, row in engineered.iterrows():
        workflow_id = row[WORKFLOW_ID_COLUMN]
        triggers = evaluate_workflow_rules(row, rule_config)

        escalation = evaluate_risk_escalation(predictions.get(workflow_id))
        if escalation is not None:
            triggers.append(escalation)

        for trigger in triggers:
            dedup_key = (workflow_id, trigger.rule_type)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            alerts.append(_alert_from_trigger(workflow_id, trigger, generated_at))

    return alerts
