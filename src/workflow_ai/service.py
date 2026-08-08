"""Pipeline orchestration: load -> validate -> predict -> KPIs -> alerts -> recommendations -> save.

run_pipeline never silently retrains a missing model -- predict.load_model
already raises a clear FileNotFoundError pointing at scripts/train_model.py.
train_if_missing is an explicit opt-in that trains and persists a model
first when one is missing, rather than the default behavior.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from workflow_ai.alert_engine import Alert, generate_alerts
from workflow_ai.config import (
    ALERTS_OUTPUT_PATH,
    DEFAULT_RULE_CONFIG,
    KPI_SUMMARY_OUTPUT_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PREDICTIONS_OUTPUT_PATH,
    RECOMMENDATIONS_OUTPUT_PATH,
    RISK_LABELS,
    RuleConfig,
)
from workflow_ai.data_loader import load_workflow_data
from workflow_ai.kpi_engine import build_kpi_summary
from workflow_ai.predict import load_model, predict_workflows
from workflow_ai.recommendations import Recommendation, generate_recommendations
from workflow_ai.validation import validate_workflow_data

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    """Everything one pipeline run produces, before writing to disk."""

    predictions: list[dict]
    kpi_summary: dict
    alerts: list[Alert]
    recommendations: list[Recommendation]


def load_and_validate_workflow_data(data_path: Path) -> pd.DataFrame:
    """Load workflow data and fail loudly if it does not pass validation."""
    df = load_workflow_data(data_path)
    validation_result = validate_workflow_data(df)
    for warning in validation_result.warnings:
        logger.warning("%s: %s", warning.check, warning.message)
    if not validation_result.is_valid:
        raise ValueError(f"Workflow data failed validation:\n{validation_result.summary()}")
    return df


def run_pipeline(
    df: pd.DataFrame,
    train_if_missing: bool = False,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
    rule_config: RuleConfig = DEFAULT_RULE_CONFIG,
) -> PipelineResult:
    """Run the full decision-support pipeline over already-loaded, validated workflow data.

    Args:
        df: Validated workflow rows.
        train_if_missing: If True and no model exists at model_path, train
            and persist one before predicting. Default False -- a missing
            model fails clearly via predict.load_model's FileNotFoundError
            rather than silently retraining.
        model_path: Where to load (and, if train_if_missing, save) the model.
        metadata_path: Where to save training metadata if train_if_missing
            trains a new model. Kept independently overridable from
            model_path so tests/callers using a scratch model_path don't
            also need to touch the real MODEL_METADATA_PATH.
        rule_config: Thresholds passed through to KPIs/alerts/recommendations.
    """
    if train_if_missing and not model_path.exists():
        logger.info("No trained model found at %s; training one now (train_if_missing=True)", model_path)
        from workflow_ai.train import save_metadata, save_model, train_and_evaluate

        training_result = train_and_evaluate(df)
        save_model(training_result.pipeline, path=model_path)
        save_metadata(training_result.metadata, path=metadata_path)

    pipeline = load_model(path=model_path)
    predictions = predict_workflows(df, pipeline=pipeline, rule_config=rule_config)

    predicted_risk_by_id = {entry["workflow_id"]: entry["predicted_risk"] for entry in predictions}
    predicted_risk_series = pd.Series([entry["predicted_risk"] for entry in predictions])

    kpi_summary = build_kpi_summary(df, risk_labels=predicted_risk_series, rule_config=rule_config)
    alerts = generate_alerts(df, predictions=predicted_risk_by_id, rule_config=rule_config)
    recommendations = generate_recommendations(df, predictions=predicted_risk_by_id, rule_config=rule_config)

    return PipelineResult(
        predictions=predictions,
        kpi_summary=kpi_summary,
        alerts=alerts,
        recommendations=recommendations,
    )


def _predictions_to_dataframe(predictions: list[dict]) -> pd.DataFrame:
    """Flatten predict_workflows's per-class risk_probability dict and risk_factors list into CSV columns."""
    columns = (
        ["workflow_id", "predicted_risk"]
        + [f"risk_probability_{label}" for label in RISK_LABELS]
        + ["risk_factors"]
    )
    rows = [
        {
            "workflow_id": entry["workflow_id"],
            "predicted_risk": entry["predicted_risk"],
            **{
                f"risk_probability_{label}": entry["risk_probability"].get(label)
                for label in RISK_LABELS
            },
            "risk_factors": "; ".join(entry["risk_factors"]),
        }
        for entry in predictions
    ]
    return pd.DataFrame(rows, columns=columns)


def _dataclasses_to_dataframe(items: list, dataclass_type: type) -> pd.DataFrame:
    """Convert a list of frozen dataclass instances to a DataFrame with a stable column order."""
    columns = [field.name for field in dataclasses.fields(dataclass_type)]
    rows = [dataclasses.asdict(item) for item in items]
    return pd.DataFrame(rows, columns=columns)


def save_outputs(
    result: PipelineResult,
    predictions_path: Path = PREDICTIONS_OUTPUT_PATH,
    alerts_path: Path = ALERTS_OUTPUT_PATH,
    kpi_summary_path: Path = KPI_SUMMARY_OUTPUT_PATH,
    recommendations_path: Path = RECOMMENDATIONS_OUTPUT_PATH,
) -> None:
    """Write a PipelineResult to the project's 4 standard output files."""
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    _predictions_to_dataframe(result.predictions).to_csv(predictions_path, index=False)
    logger.info("Wrote %d predictions to %s", len(result.predictions), predictions_path)

    alerts_path.parent.mkdir(parents=True, exist_ok=True)
    _dataclasses_to_dataframe(result.alerts, Alert).to_csv(alerts_path, index=False)
    logger.info("Wrote %d alerts to %s", len(result.alerts), alerts_path)

    kpi_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with kpi_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(result.kpi_summary, handle, indent=2)
    logger.info("Wrote KPI summary to %s", kpi_summary_path)

    recommendations_path.parent.mkdir(parents=True, exist_ok=True)
    _dataclasses_to_dataframe(result.recommendations, Recommendation).to_csv(recommendations_path, index=False)
    logger.info("Wrote %d recommendations to %s", len(result.recommendations), recommendations_path)
