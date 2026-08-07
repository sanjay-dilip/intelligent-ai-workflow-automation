"""Operational feature engineering for workflow records.

Each engineered feature has a documented operational meaning and guards
against divide-by-zero / missing denominators by returning 0.0 in that
case, rather than NaN or inf. Feature generation is deterministic: given
the same input rows, it always produces the same output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from workflow_ai.config import CATEGORICAL_FEATURE_COLUMNS, NUMERIC_FEATURE_COLUMNS

FEATURE_COLUMNS: tuple[str, ...] = tuple(NUMERIC_FEATURE_COLUMNS) + tuple(CATEGORICAL_FEATURE_COLUMNS)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Elementwise numerator/denominator, returning 0.0 where the denominator is 0 or missing."""
    numer = pd.to_numeric(numerator, errors="coerce")
    denom = pd.to_numeric(denominator, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = numer / denom
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    return ratio.fillna(0.0)


def compute_sla_utilization(df: pd.DataFrame) -> pd.Series:
    """Fraction of the SLA window consumed: elapsed_hours / sla_hours.

    A value at or above 1.0 means the SLA window has been fully consumed.
    """
    return _safe_ratio(df["elapsed_hours"], df["sla_hours"])


def compute_manual_step_ratio(df: pd.DataFrame) -> pd.Series:
    """Share of process steps that are manual: manual_steps / (manual_steps + automated_steps)."""
    total_steps = df["manual_steps"] + df["automated_steps"]
    return _safe_ratio(df["manual_steps"], total_steps)


def compute_error_rate(df: pd.DataFrame) -> pd.Series:
    """Errors per transaction: error_count / transaction_volume."""
    return _safe_ratio(df["error_count"], df["transaction_volume"])


def compute_rework_rate(df: pd.DataFrame) -> pd.Series:
    """Rework events per transaction: rework_count / transaction_volume."""
    return _safe_ratio(df["rework_count"], df["transaction_volume"])


def compute_backlog_pressure(df: pd.DataFrame) -> pd.Series:
    """Pending items relative to volume handled: pending_items / transaction_volume."""
    return _safe_ratio(df["pending_items"], df["transaction_volume"])


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with all engineered operational features added."""
    result = df.copy()
    result["sla_utilization"] = compute_sla_utilization(result)
    result["manual_step_ratio"] = compute_manual_step_ratio(result)
    result["error_rate"] = compute_error_rate(result)
    result["rework_rate"] = compute_rework_rate(result)
    result["backlog_pressure"] = compute_backlog_pressure(result)
    return result


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features and select only the model's declared feature columns.

    This is the single feature-preparation path used identically at
    training and inference time, so features are never derived
    differently between the two contexts.
    """
    engineered = engineer_features(df)
    return engineered[list(FEATURE_COLUMNS)]
