"""Centralized configuration: file paths, schema constants, and rule thresholds.

All business-rule thresholds live in RuleConfig so they are defined once and
reused by the rules engine, alert engine, and tests instead of being
scattered as magic numbers across modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Paths -------------------------------------------------------------

BASE_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = BASE_DIR / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = BASE_DIR / "models"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
OUTPUTS_PREDICTIONS_DIR: Path = OUTPUTS_DIR / "predictions"
OUTPUTS_ALERTS_DIR: Path = OUTPUTS_DIR / "alerts"
OUTPUTS_REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
EXAMPLES_DIR: Path = BASE_DIR / "examples"

MODEL_PATH: Path = MODELS_DIR / "workflow_risk_model.joblib"
MODEL_METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"

PREDICTIONS_OUTPUT_PATH: Path = OUTPUTS_PREDICTIONS_DIR / "workflow_predictions.csv"
ALERTS_OUTPUT_PATH: Path = OUTPUTS_ALERTS_DIR / "alerts.csv"
KPI_SUMMARY_OUTPUT_PATH: Path = OUTPUTS_REPORTS_DIR / "kpi_summary.json"
RECOMMENDATIONS_OUTPUT_PATH: Path = OUTPUTS_REPORTS_DIR / "recommendations.csv"

# --- Reproducibility ----------------------------------------------------

RANDOM_SEED: int = 42

# --- Dataset schema -------------------------------------------------------

WORKFLOW_ID_COLUMN: str = "workflow_id"
TARGET_COLUMN: str = "risk_label"
RISK_LABELS: tuple[str, ...] = ("low", "medium", "high")

REQUIRED_COLUMNS: tuple[str, ...] = (
    "workflow_id",
    "workflow_name",
    "department",
    "status",
    "priority",
    "sla_hours",
    "elapsed_hours",
    "manual_steps",
    "automated_steps",
    "automation_score",
    "error_count",
    "rework_count",
    "transaction_volume",
    "avg_handling_time_minutes",
    "pending_items",
    "completion_rate",
    "created_at",
    "updated_at",
)

NON_NEGATIVE_COLUMNS: tuple[str, ...] = (
    "sla_hours",
    "elapsed_hours",
    "manual_steps",
    "automated_steps",
    "error_count",
    "rework_count",
    "transaction_volume",
    "avg_handling_time_minutes",
    "pending_items",
)

# completion_rate and automation_score are stored on a 0-1 scale (not 0-100).
UNIT_INTERVAL_COLUMNS: tuple[str, ...] = ("completion_rate", "automation_score")

VALID_STATUSES: tuple[str, ...] = ("open", "in_progress", "on_hold", "completed", "cancelled")

NUMERIC_FEATURE_COLUMNS: tuple[str, ...] = (
    "sla_hours",
    "elapsed_hours",
    "manual_steps",
    "automated_steps",
    "automation_score",
    "error_count",
    "rework_count",
    "transaction_volume",
    "avg_handling_time_minutes",
    "pending_items",
    "completion_rate",
    "sla_utilization",
    "manual_step_ratio",
    "error_rate",
    "rework_rate",
    "backlog_pressure",
)

CATEGORICAL_FEATURE_COLUMNS: tuple[str, ...] = ("department", "status", "priority")

# Columns that must never be used as model features because they leak the
# target or reflect post-hoc/administrative information, not predictive signal.
LEAKAGE_COLUMNS: tuple[str, ...] = (
    "risk_label",
    "workflow_id",
    "workflow_name",
    "created_at",
    "updated_at",
)


@dataclass(frozen=True)
class RuleConfig:
    """Centralized, deterministic thresholds for the rules and alert engines."""

    sla_warning_threshold: float = 0.80
    high_error_threshold: int = 5
    high_rework_rate: float = 0.10
    high_manual_ratio: float = 0.70
    low_automation_threshold: float = 0.30
    backlog_threshold: int = 100


DEFAULT_RULE_CONFIG = RuleConfig()
