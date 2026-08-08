"""Inference over the persisted risk model plus deterministic risk-factor explanations.

risk_factors returned here are threshold-based checks over a workflow's
own operational fields (reusing RuleConfig), not an explanation of the
Random Forest's internal feature importances -- individual predictions
are never explained by aggregate feature importance, since that is a
property of the whole trained model, not of any one row.

risk_factor_fn defaults to rules_engine.risk_factors_from_rules, the
project's single source of truth for rule thresholds. It remains
injectable so callers/tests can override it without changing
predict_workflows itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from workflow_ai.config import DEFAULT_RULE_CONFIG, MODEL_PATH, RuleConfig, WORKFLOW_ID_COLUMN
from workflow_ai.feature_engineering import engineer_features, prepare_feature_frame
from workflow_ai.rules_engine import risk_factors_from_rules

RiskFactorFn = Callable[[pd.Series, RuleConfig], list[str]]


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    """Load the persisted, fitted risk model pipeline.

    Raises:
        FileNotFoundError: If no trained model exists at ``path``.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Run scripts/train_model.py first."
        )
    return joblib.load(path)


def predict_workflows(
    df: pd.DataFrame,
    pipeline: Pipeline | None = None,
    risk_factor_fn: RiskFactorFn = risk_factors_from_rules,
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
            str(cls): float(probabilities[position][class_index])
            for class_index, cls in enumerate(classes)
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
