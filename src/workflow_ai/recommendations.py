"""Category-tagged recommendations derived from rules_engine triggers.

Every recommendation is tied to an observed signal (a fired rule, or an
explicit "nothing triggered" state informed by the model's predicted
risk) -- never a claim not backed by the row's own data.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from workflow_ai.config import DEFAULT_RULE_CONFIG, RuleConfig, WORKFLOW_ID_COLUMN
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
)

ESCALATE = "escalate"
REVIEW_SLA = "review_sla"
INVESTIGATE_ERRORS = "investigate_errors"
REDUCE_REWORK = "reduce_rework"
ASSESS_AUTOMATION = "assess_automation"
REBALANCE_WORKLOAD = "rebalance_workload"
MONITOR = "monitor"
NO_ACTION = "no_action"

_CATEGORY_BY_RULE_TYPE: dict[str, str] = {
    SLA_BREACH: ESCALATE,
    HIGH_RISK_ESCALATION: ESCALATE,
    APPROACHING_SLA: REVIEW_SLA,
    HIGH_ERROR: INVESTIGATE_ERRORS,
    HIGH_REWORK: REDUCE_REWORK,
    MANUAL_AUTOMATION_OPPORTUNITY: ASSESS_AUTOMATION,
    BACKLOG_PRESSURE: REBALANCE_WORKLOAD,
}

_MEDIUM_RISK_NO_TRIGGERS_MESSAGE = (
    "No rule thresholds triggered, but the model predicts medium risk"
)
_NO_SIGNAL_MESSAGE = "No rule thresholds triggered and no elevated predicted risk"


@dataclass(frozen=True)
class Recommendation:
    """A recommended action for one workflow, tied to an observed signal."""

    workflow_id: str
    category: str
    message: str
    signal: str


def generate_recommendations_for_workflow(
    workflow_id: str,
    row: pd.Series,
    predicted_risk: str | None = None,
    rule_config: RuleConfig = DEFAULT_RULE_CONFIG,
) -> list[Recommendation]:
    """Generate recommendations for one engineered workflow row."""
    triggers = evaluate_workflow_rules(row, rule_config)

    escalation = evaluate_risk_escalation(predicted_risk)
    if escalation is not None:
        triggers.append(escalation)

    if triggers:
        return [
            Recommendation(
                workflow_id=workflow_id,
                category=_CATEGORY_BY_RULE_TYPE[trigger.rule_type],
                message=trigger.recommended_action,
                signal=trigger.message,
            )
            for trigger in triggers
        ]

    if predicted_risk == "medium":
        return [
            Recommendation(
                workflow_id=workflow_id,
                category=MONITOR,
                message="Continue monitoring; no immediate action required",
                signal=_MEDIUM_RISK_NO_TRIGGERS_MESSAGE,
            )
        ]

    return [
        Recommendation(
            workflow_id=workflow_id,
            category=NO_ACTION,
            message="No action needed at this time",
            signal=_NO_SIGNAL_MESSAGE,
        )
    ]


def generate_recommendations(
    df: pd.DataFrame,
    predictions: dict[str, str] | None = None,
    rule_config: RuleConfig = DEFAULT_RULE_CONFIG,
) -> list[Recommendation]:
    """Generate recommendations for every workflow row in df."""
    predictions = predictions or {}
    engineered = engineer_features(df)

    recommendations: list[Recommendation] = []
    for _, row in engineered.iterrows():
        workflow_id = row[WORKFLOW_ID_COLUMN]
        recommendations.extend(
            generate_recommendations_for_workflow(
                workflow_id, row, predictions.get(workflow_id), rule_config
            )
        )
    return recommendations
