"""Deterministic KPI aggregation over workflow records.

Every KPI here is a plain, documented formula over the raw or engineered
workflow columns -- no model output is required. Functions that need
sla_utilization/manual_step_ratio/etc. call engineer_features
internally, the same shared path used elsewhere in the project.
"""

from __future__ import annotations

import pandas as pd

from workflow_ai.config import DEFAULT_RULE_CONFIG, RISK_LABELS, RuleConfig
from workflow_ai.feature_engineering import engineer_features
from workflow_ai.rules_engine import SLA_BREACH_UTILIZATION


def compute_volume_kpis(df: pd.DataFrame) -> dict:
    """Workflow counts overall and broken down by status/department/priority."""
    return {
        "total_workflows": len(df),
        "by_status": df["status"].value_counts().to_dict(),
        "by_department": df["department"].value_counts().to_dict(),
        "by_priority": df["priority"].value_counts().to_dict(),
    }


def compute_sla_kpis(df: pd.DataFrame, rule_config: RuleConfig = DEFAULT_RULE_CONFIG) -> dict:
    """SLA health: average utilization, breach rate, and approaching-breach rate.

    A breach is sla_utilization >= SLA_BREACH_UTILIZATION (the full SLA
    window consumed). "Approaching" is at or above the warning threshold
    but not yet breached, mirroring rules_engine's mutually-exclusive
    breach/approaching split.
    """
    engineered = engineer_features(df)
    utilization = engineered["sla_utilization"]
    total = len(engineered)

    breach_mask = utilization >= SLA_BREACH_UTILIZATION
    approaching_mask = (utilization >= rule_config.sla_warning_threshold) & ~breach_mask

    return {
        "avg_sla_utilization": float(utilization.mean()) if total else 0.0,
        "sla_breach_count": int(breach_mask.sum()),
        "sla_breach_rate": float(breach_mask.sum() / total) if total else 0.0,
        "sla_approaching_count": int(approaching_mask.sum()),
        "sla_approaching_rate": float(approaching_mask.sum() / total) if total else 0.0,
    }


def compute_efficiency_kpis(df: pd.DataFrame) -> dict:
    """Average operational efficiency signals across all workflows."""
    engineered = engineer_features(df)
    return {
        "avg_manual_step_ratio": float(engineered["manual_step_ratio"].mean()),
        "avg_automation_score": float(engineered["automation_score"].mean()),
        "avg_completion_rate": float(engineered["completion_rate"].mean()),
        "avg_handling_time_minutes": float(engineered["avg_handling_time_minutes"].mean()),
        "avg_error_rate": float(engineered["error_rate"].mean()),
        "avg_rework_rate": float(engineered["rework_rate"].mean()),
    }


def compute_risk_distribution(risk_labels: pd.Series) -> dict:
    """Count and share of workflows per risk label, in RISK_LABELS order.

    Args:
        risk_labels: A Series of "low"/"medium"/"high" values -- either
            the synthetic risk_label column (for historical/demo
            reporting) or model-predicted risk, depending on caller.
    """
    total = len(risk_labels)
    counts = risk_labels.value_counts().reindex(RISK_LABELS, fill_value=0).astype(int)
    shares = (counts / total) if total else counts.astype(float)
    return {
        "counts": counts.to_dict(),
        "shares": shares.to_dict(),
    }


def compute_department_breakdown(df: pd.DataFrame) -> list[dict]:
    """Per-department workflow count and average SLA utilization / error rate."""
    engineered = engineer_features(df)
    grouped = engineered.groupby("department", observed=True).agg(
        workflow_count=("department", "count"),
        avg_sla_utilization=("sla_utilization", "mean"),
        avg_error_rate=("error_rate", "mean"),
    )
    return [
        {
            "department": department,
            "workflow_count": int(row["workflow_count"]),
            "avg_sla_utilization": float(row["avg_sla_utilization"]),
            "avg_error_rate": float(row["avg_error_rate"]),
        }
        for department, row in grouped.iterrows()
    ]


def build_kpi_summary(
    df: pd.DataFrame,
    risk_labels: pd.Series | None = None,
    rule_config: RuleConfig = DEFAULT_RULE_CONFIG,
) -> dict:
    """Assemble the full KPI summary: volume, SLA, efficiency, risk, department breakdown."""
    summary = {
        "volume": compute_volume_kpis(df),
        "sla": compute_sla_kpis(df, rule_config),
        "efficiency": compute_efficiency_kpis(df),
        "by_department": compute_department_breakdown(df),
    }
    if risk_labels is not None:
        summary["risk"] = compute_risk_distribution(risk_labels)
    return summary
