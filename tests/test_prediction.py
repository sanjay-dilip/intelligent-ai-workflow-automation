"""Tests for inference and deterministic risk-factor explanations."""

from __future__ import annotations

import pandas as pd
import pytest

from workflow_ai.config import RISK_LABELS
from workflow_ai.models import build_random_forest_pipeline
from workflow_ai.predict import load_model, predict_workflows
from workflow_ai.rules_engine import risk_factors_from_rules

BASE_ROW = {
    "workflow_id": "WF-1000",
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
    "risk_label": "low",
}


def _df(overrides: dict | None = None) -> pd.DataFrame:
    row = dict(BASE_ROW)
    if overrides:
        row.update(overrides)
    return pd.DataFrame([row])


def _fitted_pipeline():
    rows = []
    for label, bias in (("low", 0.1), ("medium", 0.5), ("high", 0.9)):
        for i in range(10):
            rows.append(
                _df(
                    {
                        "workflow_id": f"WF-{label}-{i}",
                        "elapsed_hours": 100.0 * bias,
                        "error_count": int(bias * 10),
                        "automation_score": 1 - bias,
                        "risk_label": label,
                    }
                ).iloc[0]
            )
    train_df = pd.DataFrame(rows)
    pipeline = build_random_forest_pipeline()
    from workflow_ai.feature_engineering import prepare_feature_frame

    pipeline.fit(prepare_feature_frame(train_df), train_df["risk_label"])
    return pipeline


def test_predict_workflows_returns_one_dict_per_row() -> None:
    pipeline = _fitted_pipeline()
    df = pd.concat([_df(), _df({"workflow_id": "WF-1001"})], ignore_index=True)

    results = predict_workflows(df, pipeline=pipeline)

    assert len(results) == len(df)
    for entry in results:
        assert set(entry.keys()) == {
            "workflow_id",
            "predicted_risk",
            "risk_probability",
            "risk_factors",
        }


def test_predict_workflows_predicted_risk_is_valid_label() -> None:
    pipeline = _fitted_pipeline()
    results = predict_workflows(_df(), pipeline=pipeline)
    assert results[0]["predicted_risk"] in RISK_LABELS


def test_predict_workflows_probabilities_sum_to_one_and_cover_all_classes() -> None:
    pipeline = _fitted_pipeline()
    results = predict_workflows(_df(), pipeline=pipeline)
    probability = results[0]["risk_probability"]
    assert set(probability.keys()) == set(RISK_LABELS)
    assert sum(probability.values()) == pytest.approx(1.0)


def test_predict_workflows_defaults_to_rules_engine_risk_factors() -> None:
    pipeline = _fitted_pipeline()
    df = _df({"elapsed_hours": 95.0})

    default_result = predict_workflows(df, pipeline=pipeline)[0]
    explicit_result = predict_workflows(
        df, pipeline=pipeline, risk_factor_fn=risk_factors_from_rules
    )[0]

    assert default_result["risk_factors"] == explicit_result["risk_factors"]
    assert any("SLA" in factor for factor in default_result["risk_factors"])


def test_predict_workflows_ignores_leakage_columns_in_model_input() -> None:
    pipeline = _fitted_pipeline()
    with_leakage = _df({"workflow_id": "WF-9999", "risk_label": "high"})
    without_leakage = with_leakage.drop(columns=["workflow_id", "risk_label"])

    with_leakage_result = predict_workflows(with_leakage, pipeline=pipeline)[0]
    without_leakage_result = predict_workflows(without_leakage, pipeline=pipeline)[0]

    assert with_leakage_result["predicted_risk"] == without_leakage_result["predicted_risk"]
    assert with_leakage_result["risk_probability"] == without_leakage_result["risk_probability"]


def test_load_model_raises_file_not_found_when_missing(tmp_path) -> None:
    missing_path = tmp_path / "does_not_exist.joblib"
    with pytest.raises(FileNotFoundError):
        load_model(path=missing_path)
