"""Tests for pipeline orchestration: run_pipeline, save_outputs, and the CLI's data loading."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from workflow_ai.config import RISK_LABELS
from workflow_ai.feature_engineering import prepare_feature_frame
from workflow_ai.models import build_random_forest_pipeline
from workflow_ai.service import load_and_validate_workflow_data, run_pipeline, save_outputs
from workflow_ai.train import save_model

BASE_ROW = {
    "workflow_name": "Invoice Processing",
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
    "created_at": pd.Timestamp("2026-01-01"),
    "updated_at": pd.Timestamp("2026-01-02"),
}


def _synthetic_dataset() -> pd.DataFrame:
    rows = []
    for label, bias in (("low", 0.1), ("medium", 0.5), ("high", 0.9)):
        for i in range(10):
            row = dict(BASE_ROW)
            row.update(
                {
                    "workflow_id": f"WF-{label}-{i}",
                    "elapsed_hours": 100.0 * bias,
                    "error_count": int(bias * 10),
                    "automation_score": 1 - bias,
                    "pending_items": int(bias * 150),
                    "risk_label": label,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _fitted_model_path(tmp_path):
    df = _synthetic_dataset()
    pipeline = build_random_forest_pipeline()
    pipeline.fit(prepare_feature_frame(df), df["risk_label"])
    model_path = tmp_path / "model.joblib"
    save_model(pipeline, path=model_path)
    return model_path


def test_run_pipeline_produces_predictions_kpis_alerts_recommendations(tmp_path) -> None:
    df = _synthetic_dataset()
    model_path = _fitted_model_path(tmp_path)

    result = run_pipeline(df, model_path=model_path)

    assert len(result.predictions) == len(df)
    assert set(result.kpi_summary.keys()) == {"volume", "sla", "efficiency", "by_department", "risk"}
    assert isinstance(result.alerts, list)
    # Each workflow yields at least one recommendation (one per triggered rule, or a fallback).
    assert len(result.recommendations) >= len(df)
    assert set(rec.workflow_id for rec in result.recommendations) == set(df["workflow_id"])


def test_run_pipeline_raises_clearly_when_model_missing(tmp_path) -> None:
    df = _synthetic_dataset()
    missing_model_path = tmp_path / "does_not_exist.joblib"

    with pytest.raises(FileNotFoundError):
        run_pipeline(df, model_path=missing_model_path)


def test_run_pipeline_trains_when_missing_and_opted_in(tmp_path) -> None:
    df = _synthetic_dataset()
    missing_model_path = tmp_path / "does_not_exist.joblib"
    metadata_path = tmp_path / "metadata.json"

    result = run_pipeline(df, train_if_missing=True, model_path=missing_model_path, metadata_path=metadata_path)

    assert missing_model_path.exists()
    assert metadata_path.exists()
    assert len(result.predictions) == len(df)


def test_save_outputs_writes_all_four_files_with_expected_schema(tmp_path) -> None:
    df = _synthetic_dataset()
    model_path = _fitted_model_path(tmp_path)
    result = run_pipeline(df, model_path=model_path)

    predictions_path = tmp_path / "predictions.csv"
    alerts_path = tmp_path / "alerts.csv"
    kpi_summary_path = tmp_path / "kpi_summary.json"
    recommendations_path = tmp_path / "recommendations.csv"

    save_outputs(
        result,
        predictions_path=predictions_path,
        alerts_path=alerts_path,
        kpi_summary_path=kpi_summary_path,
        recommendations_path=recommendations_path,
    )

    predictions_df = pd.read_csv(predictions_path)
    assert len(predictions_df) == len(df)
    expected_prediction_columns = (
        {"workflow_id", "predicted_risk", "risk_factors"} | {f"risk_probability_{label}" for label in RISK_LABELS}
    )
    assert expected_prediction_columns.issubset(set(predictions_df.columns))

    alerts_df = pd.read_csv(alerts_path)
    assert list(alerts_df.columns) == [
        "alert_id",
        "workflow_id",
        "severity",
        "type",
        "message",
        "recommended_action",
        "generated_at",
    ]

    with kpi_summary_path.open(encoding="utf-8") as handle:
        kpi_summary = json.load(handle)
    assert kpi_summary == result.kpi_summary

    recommendations_df = pd.read_csv(recommendations_path)
    assert list(recommendations_df.columns) == ["workflow_id", "category", "message", "signal"]
    assert len(recommendations_df) == len(result.recommendations)


def test_save_outputs_writes_empty_alerts_file_with_headers_when_no_alerts(tmp_path) -> None:
    df = pd.DataFrame(
        [
            {
                **BASE_ROW,
                "workflow_id": "WF-safe",
                "risk_label": "low",
            }
        ]
    )
    model_path = _fitted_model_path(tmp_path)
    result = run_pipeline(df, model_path=model_path)

    alerts_path = tmp_path / "alerts.csv"
    save_outputs(
        result,
        predictions_path=tmp_path / "predictions.csv",
        alerts_path=alerts_path,
        kpi_summary_path=tmp_path / "kpi_summary.json",
        recommendations_path=tmp_path / "recommendations.csv",
    )

    alerts_df = pd.read_csv(alerts_path)
    assert list(alerts_df.columns) == [
        "alert_id",
        "workflow_id",
        "severity",
        "type",
        "message",
        "recommended_action",
        "generated_at",
    ]
    assert len(alerts_df) == 0


def test_load_and_validate_workflow_data_raises_on_invalid_data(tmp_path) -> None:
    invalid_df = pd.DataFrame([{**BASE_ROW, "workflow_id": "WF-1", "risk_label": "low", "sla_hours": -5.0}])
    data_path = tmp_path / "invalid.csv"
    invalid_df.to_csv(data_path, index=False)

    with pytest.raises(ValueError):
        load_and_validate_workflow_data(data_path)


def test_load_and_validate_workflow_data_returns_df_for_valid_data(tmp_path) -> None:
    valid_df = _synthetic_dataset()
    data_path = tmp_path / "valid.csv"
    valid_df.to_csv(data_path, index=False)

    loaded = load_and_validate_workflow_data(data_path)
    assert len(loaded) == len(valid_df)
