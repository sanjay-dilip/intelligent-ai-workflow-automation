"""Deterministic rules engine for workflow risk factors.

Every threshold used here comes from RuleConfig/DEFAULT_RULE_CONFIG so
thresholds are defined once and never scattered as magic numbers. This
module implements the 7 rules from the project spec: SLA breach,
approaching SLA, high error, rework, manual-process/automation
opportunity, backlog pressure, and high-risk escalation.

risk_factors_from_rules() matches predict.py's RiskFactorFn signature
and is the seam left open in Build 4: repointing predict_workflows's
risk_factor_fn default to this function replaces Build 4's standalone
placeholder logic with this rules engine, with no other code changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from workflow_ai.config import DEFAULT_RULE_CONFIG, RuleConfig

# A workflow's SLA window is considered fully consumed (breached) at or
# above this fraction, distinct from RuleConfig.sla_warning_threshold
# which flags an approaching-but-not-yet-breached SLA.
SLA_BREACH_UTILIZATION: float = 1.0

_NO_TRIGGERS_MESSAGE: str = "No individual risk thresholds exceeded"

SLA_BREACH = "sla_breach"
APPROACHING_SLA = "approaching_sla"
HIGH_ERROR = "high_error"
HIGH_REWORK = "high_rework"
MANUAL_AUTOMATION_OPPORTUNITY = "manual_automation_opportunity"
BACKLOG_PRESSURE = "backlog_pressure"
HIGH_RISK_ESCALATION = "high_risk_escalation"

ALL_RULE_TYPES: tuple[str, ...] = (
    SLA_BREACH,
    APPROACHING_SLA,
    HIGH_ERROR,
    HIGH_REWORK,
    MANUAL_AUTOMATION_OPPORTUNITY,
    BACKLOG_PRESSURE,
    HIGH_RISK_ESCALATION,
)

_RECOMMENDED_ACTIONS: dict[str, str] = {
    SLA_BREACH: "Escalate to the process owner and expedite remaining steps immediately",
    APPROACHING_SLA: "Monitor closely and prioritize remaining steps before the SLA is breached",
    HIGH_ERROR: "Investigate the root cause of the elevated error count",
    HIGH_REWORK: "Review rework triggers and quality checkpoints",
    MANUAL_AUTOMATION_OPPORTUNITY: "Assess this workflow for automation to reduce manual effort",
    BACKLOG_PRESSURE: "Rebalance workload to reduce the pending item backlog",
    HIGH_RISK_ESCALATION: "Escalate to an operations manager for review",
}


@dataclass(frozen=True)
class RuleTrigger:
    """One rule firing for one workflow row."""

    rule_type: str
    severity: str
    message: str
    recommended_action: str


def evaluate_workflow_rules(row: pd.Series, rule_config: RuleConfig = DEFAULT_RULE_CONFIG) -> list[RuleTrigger]:
    """Evaluate the 6 row-level rules against one engineered workflow row.

    Row-level rules only -- high-risk escalation (the 7th rule) depends
    on the model's predicted_risk, not on row fields, and is evaluated
    separately by evaluate_risk_escalation.

    SLA breach and approaching-SLA are mutually exclusive: a fully
    breached SLA (>= SLA_BREACH_UTILIZATION) is reported as a breach,
    not also as "approaching".
    """
    triggers: list[RuleTrigger] = []

    sla_utilization = row["sla_utilization"]
    if sla_utilization >= SLA_BREACH_UTILIZATION:
        triggers.append(
            RuleTrigger(
                rule_type=SLA_BREACH,
                severity="critical",
                message=f"SLA breached: utilization at {sla_utilization:.0%} of the allotted window",
                recommended_action=_RECOMMENDED_ACTIONS[SLA_BREACH],
            )
        )
    elif sla_utilization >= rule_config.sla_warning_threshold:
        triggers.append(
            RuleTrigger(
                rule_type=APPROACHING_SLA,
                severity="warning",
                message=(
                    f"SLA utilization {sla_utilization:.0%} at or above the "
                    f"{rule_config.sla_warning_threshold:.0%} warning threshold"
                ),
                recommended_action=_RECOMMENDED_ACTIONS[APPROACHING_SLA],
            )
        )

    if row["error_count"] >= rule_config.high_error_threshold:
        triggers.append(
            RuleTrigger(
                rule_type=HIGH_ERROR,
                severity="warning",
                message=f"Error count {int(row['error_count'])} at or above threshold {rule_config.high_error_threshold}",
                recommended_action=_RECOMMENDED_ACTIONS[HIGH_ERROR],
            )
        )

    if row["rework_rate"] >= rule_config.high_rework_rate:
        triggers.append(
            RuleTrigger(
                rule_type=HIGH_REWORK,
                severity="warning",
                message=(
                    f"Rework rate {row['rework_rate']:.0%} at or above the "
                    f"{rule_config.high_rework_rate:.0%} threshold"
                ),
                recommended_action=_RECOMMENDED_ACTIONS[HIGH_REWORK],
            )
        )

    if row["manual_step_ratio"] >= rule_config.high_manual_ratio or row["automation_score"] <= rule_config.low_automation_threshold:
        triggers.append(
            RuleTrigger(
                rule_type=MANUAL_AUTOMATION_OPPORTUNITY,
                severity="info",
                message=(
                    f"Manual step ratio {row['manual_step_ratio']:.0%} / automation score "
                    f"{row['automation_score']:.0%} indicate an automation opportunity"
                ),
                recommended_action=_RECOMMENDED_ACTIONS[MANUAL_AUTOMATION_OPPORTUNITY],
            )
        )

    if row["pending_items"] >= rule_config.backlog_threshold:
        triggers.append(
            RuleTrigger(
                rule_type=BACKLOG_PRESSURE,
                severity="warning",
                message=(
                    f"Pending items {int(row['pending_items'])} at or above backlog "
                    f"threshold {rule_config.backlog_threshold}"
                ),
                recommended_action=_RECOMMENDED_ACTIONS[BACKLOG_PRESSURE],
            )
        )

    return triggers


def evaluate_risk_escalation(predicted_risk: str | None) -> RuleTrigger | None:
    """Evaluate the 7th rule: escalate when the model's predicted risk is "high"."""
    if predicted_risk != "high":
        return None
    return RuleTrigger(
        rule_type=HIGH_RISK_ESCALATION,
        severity="critical",
        message="Model predicts high risk for this workflow",
        recommended_action=_RECOMMENDED_ACTIONS[HIGH_RISK_ESCALATION],
    )


def risk_factors_from_rules(row: pd.Series, rule_config: RuleConfig = DEFAULT_RULE_CONFIG) -> list[str]:
    """Adapter matching predict.py's RiskFactorFn signature.

    Repoint predict_workflows's risk_factor_fn default to this function
    to replace Build 4's standalone placeholder logic with this rules
    engine's row-level rules, without changing predict_workflows itself.
    """
    triggers = evaluate_workflow_rules(row, rule_config)
    if not triggers:
        return [_NO_TRIGGERS_MESSAGE]
    return [trigger.message for trigger in triggers]
