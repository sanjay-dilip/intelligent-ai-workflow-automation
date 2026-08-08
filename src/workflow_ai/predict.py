"""Inference over the persisted risk model plus deterministic risk-factor explanations.

risk_factors returned here are threshold-based checks over a workflow's
own operational fields (reusing RuleConfig), not an explanation of the
Random Forest's internal feature importances -- individual predictions
are never explained by aggregate feature importance, since that is a
property of the whole trained model, not of any one row.

The default risk-factor logic below is intentionally small and
standalone. It is injectable via the risk_factor_fn parameter so a
future, richer rules engine (see the project plan's Build 5) can
supersede it without changing predict_workflows itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from workflow_ai.config import DEFAULT_RULE_CONFIG, MODEL_PATH, RuleConfig, WORKFLOW_ID_COLUMN
from workflow_ai.feature_engineering import engineer_features, prepare_feature_frame

RiskFactorFn = Callable[[pd.Series, RuleConfig], list[str]]

_NO_RISK_FACTORS_MESSAGE: str = "No individual risk thresholds exceeded"


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    """Load the persisted, fitted risk model pipeline.

    Raises:
        FileNotFoundError: If no trained model exists at ``path``.
    """
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found at {path}. Run scripts/train_model.py first.")
    return joblib.load(path)


def _default_risk_factors(row: pd.Series, rule_config: RuleConfig = DEFAULT_RULE_CONFIG) -> list[str]:
    """Deterministic, threshold-based explanation for one engineered workflow row."""
    factors: list[str] = []

    if row["sla_utilization"] >= rule_config.sla_warning_threshold:
        factors.append(
            f"SLA utilization {row['sla_utilization']:.0%} at or above the "
            f"{rule_config.sla_warning_threshold:.0%} warning threshold"
        )
    if row["error_count"] >= rule_config.high_error_threshold:
        factors.append(f"Error count {int(row['error_count'])} at or above threshold {rule_config.high_error_threshold}")
    if row["rework_rate"] >= rule_config.high_rework_rate:
        factors.append(
            f"Rework rate {row['rework_rate']:.0%} at or above the {rule_config.high_rework_rate:.0%} threshold"
        )
    if row["manual_step_ratio"] >= rule_config.high_manual_ratio:
        factors.append(
            f"Manual step ratio {row['manual_step_ratio']:.0%} at or above the "
            f"{rule_config.high_manual_ratio:.0%} threshold"
        )
    if row["automation_score"] <= rule_config.low_automation_threshold:
        factors.append(
            f"Automation score {row['automation_score']:.0%} at or below the "
            f"{rule_config.low_automation_threshold:.0%} threshold"
        )
    if row["pending_items"] >= rule_config.backlog_threshold:
        factors.append(
            f"Pending items {int(row['pending_items'])} at or above backlog threshold {rule_config.backlog_threshold}"
        )

    if not factors:
        factors.append(_NO_RISK_FACTORS_MESSAGE)
    return factors


def predict_workflows(
    df: pd.DataFrame,
    pipeline: Pipeline | None = None,
    risk_factor_fn: RiskFactorFn = _default_risk_factors,
    rule_config: RuleConfig = DEFAULT_RULE_CONFIG,
) -> list[dict]:
    """Predict risk for each workflow row and attach a deterministic explanation.

    Args:
        df: Raw workflow rows (any leakage columns present, e.g. workflow_id
            or risk_label, are ignored by feature preparation, not passed
            to the model).
        pipeline: A fitted pipeline to use instead of loading MODEL_PATH.
            Lets tests inject an in-memory pipeline with no disk I/O.
        risk_factor_fn: Function producing the risk_factors list for one
            engineered row; overridable seam for a future rules engine.
        rule_config: Thresholds passed through to risk_factor_fn.

    Returns:
        One dict per input row: workflow_id, predicted_risk,
        risk_probability (per-class dict), risk_factors.
    """
    if pipeline is None:
        pipeline = load_model()

    features = prepare_feature_frame(df)
    engineered = engineer_features(df)

    predicted_labels = pipeline.predict(features)
    probabilities = pipeline.predict_proba(features)
    classes = list(pipeline.classes_)

    results: list[dict] = []
    for position, (_, engineered_row) in enumerate(engineered.iterrows()):
        risk_probability = {
            str(cls): float(probabilities[position][class_index]) for class_index, cls in enumerate(classes)
        }
        results.append(
            {
                "workflow_id": engineered_row.get(WORKFLOW_ID_COLUMN),
                "predicted_risk": predicted_labels[position],
                "risk_probability": risk_probability,
                "risk_factors": risk_factor_fn(engineered_row, rule_config),
            }
        )
    return results
