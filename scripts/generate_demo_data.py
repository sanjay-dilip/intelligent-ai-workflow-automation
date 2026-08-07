"""Generate a synthetic, clearly-labeled demonstration workflow dataset.

*** THIS DATA IS SYNTHETIC. *** It does not represent any real business
process and exists only to demonstrate and test the workflow-risk pipeline
end to end. Model performance measured on this data does not indicate
real-world performance.

Risk labels are assigned probabilistically from a noisy weighted
combination of several operational signals (SLA utilization, error rate,
rework rate, manual-step ratio, automation score, completion rate, backlog
pressure), then sampled from a soft, overlapping per-class distribution
rather than a hard cutoff. This keeps the label generation independent of
the hard per-field thresholds the rules engine applies later (avoiding
trivial train/rules leakage) and leaves genuine class overlap so the
classification task is not perfectly separable.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from workflow_ai.config import DATA_RAW_DIR, EXAMPLES_DIR, RANDOM_SEED, RISK_LABELS

logger = logging.getLogger(__name__)

DEPARTMENTS = (
    "Finance",
    "IT",
    "HR",
    "Operations",
    "Customer Support",
    "Procurement",
    "Compliance",
)
WORKFLOW_TYPES = (
    "Invoice Processing",
    "Purchase Order Approval",
    "Customer Support Escalation",
    "Employee Onboarding",
    "Vendor Approval",
    "Claims Processing",
    "Order Fulfillment",
    "IT Service Request",
    "Compliance Review",
)
STATUSES = ("open", "in_progress", "on_hold", "completed", "cancelled")
STATUS_PROBABILITIES = (0.25, 0.35, 0.10, 0.25, 0.05)
PRIORITIES = ("low", "medium", "high", "urgent")
PRIORITY_PROBABILITIES = (0.30, 0.40, 0.20, 0.10)

# Soft-label sampling: each risk class has a center on the [0, 1] risk-score
# axis; temperature controls how much class probability mass overlaps
# between neighboring classes (higher = more overlap/noise).
_RISK_CLASS_CENTERS = {"low": 0.15, "medium": 0.50, "high": 0.85}
_RISK_LABEL_TEMPERATURE = 0.35


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _assign_soft_risk_labels(risk_score: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample a risk_label per row from a soft, overlapping class distribution.

    Each row's label is sampled from a categorical distribution derived
    from a temperature-scaled distance between the row's risk score and
    each class center, producing genuine class overlap instead of a hard
    threshold cutoff.
    """
    logits = np.stack(
        [-np.abs(risk_score - _RISK_CLASS_CENTERS[label]) / _RISK_LABEL_TEMPERATURE for label in RISK_LABELS],
        axis=1,
    )
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    labels = np.array(RISK_LABELS)
    return np.array([rng.choice(labels, p=probabilities[i]) for i in range(len(risk_score))])


def generate_demo_dataset(n_rows: int = 2000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a synthetic workflow dataset with a probabilistic risk label.

    Args:
        n_rows: Number of workflow records to generate.
        seed: Random seed for full reproducibility.

    Returns:
        A DataFrame matching the project's workflow schema, including a
        synthetic ``risk_label`` column.
    """
    rng = np.random.default_rng(seed)

    workflow_id = [f"WF-{1000 + i}" for i in range(n_rows)]
    department = rng.choice(DEPARTMENTS, size=n_rows)
    workflow_name = rng.choice(WORKFLOW_TYPES, size=n_rows)
    status = rng.choice(STATUSES, size=n_rows, p=STATUS_PROBABILITIES)
    priority = rng.choice(PRIORITIES, size=n_rows, p=PRIORITY_PROBABILITIES)

    # Latent per-row risk propensity drives correlated, noisy operational signals.
    propensity = rng.beta(2, 2, size=n_rows)

    sla_hours = rng.uniform(4, 168, size=n_rows)
    sla_utilization_raw = np.clip(0.2 + 0.9 * propensity + rng.normal(0, 0.15, size=n_rows), 0.01, 1.6)
    elapsed_hours = sla_hours * sla_utilization_raw

    total_steps = rng.integers(3, 25, size=n_rows)
    manual_ratio_raw = np.clip(0.15 + 0.7 * propensity + rng.normal(0, 0.1, size=n_rows), 0.0, 1.0)
    manual_steps = np.round(total_steps * manual_ratio_raw).astype(int)
    automated_steps = total_steps - manual_steps

    automation_score = np.clip(0.9 - 0.7 * propensity + rng.normal(0, 0.1, size=n_rows), 0.0, 1.0)

    transaction_volume = np.clip(np.round(rng.lognormal(mean=4.5, sigma=0.8, size=n_rows)).astype(int), 1, None)

    error_count = rng.poisson(0.5 + 6 * propensity, size=n_rows)
    rework_count = rng.poisson(0.3 + 4 * propensity, size=n_rows)

    avg_handling_time_minutes = np.clip(15 + 90 * manual_ratio_raw + rng.normal(0, 10, size=n_rows), 5, None)
    pending_items = rng.poisson(10 + 150 * propensity, size=n_rows)
    completion_rate = np.clip(0.95 - 0.5 * propensity + rng.normal(0, 0.08, size=n_rows), 0.0, 1.0)

    created_at = pd.Timestamp.now().normalize() - pd.to_timedelta(rng.integers(0, 90, size=n_rows), unit="D")
    updated_at = created_at + pd.to_timedelta(elapsed_hours, unit="h")

    error_rate = error_count / transaction_volume
    rework_rate = rework_count / transaction_volume
    backlog_pressure = pending_items / transaction_volume

    risk_score = _sigmoid(
        2.2 * (sla_utilization_raw - 0.7)
        + 1.8 * (manual_ratio_raw - 0.5)
        + 1.5 * (error_rate * 5)
        + 1.5 * (rework_rate * 5)
        + 1.2 * (backlog_pressure * 2)
        - 1.5 * (automation_score - 0.5)
        - 1.0 * (completion_rate - 0.5)
        + rng.normal(0, 0.35, size=n_rows)
    )
    risk_label = _assign_soft_risk_labels(risk_score, rng)

    return pd.DataFrame(
        {
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
            "department": department,
            "status": status,
            "priority": priority,
            "sla_hours": np.round(sla_hours, 2),
            "elapsed_hours": np.round(elapsed_hours, 2),
            "manual_steps": manual_steps,
            "automated_steps": automated_steps,
            "automation_score": np.round(automation_score, 3),
            "error_count": error_count,
            "rework_count": rework_count,
            "transaction_volume": transaction_volume,
            "avg_handling_time_minutes": np.round(avg_handling_time_minutes, 1),
            "pending_items": pending_items,
            "completion_rate": np.round(completion_rate, 3),
            "created_at": created_at,
            "updated_at": updated_at,
            "risk_label": risk_label,
        }
    )


def main() -> None:
    """CLI entry point: write a full training dataset and a small checked-in sample."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Generate synthetic demo workflow data.")
    parser.add_argument("--rows", type=int, default=2000, help="Number of rows for the full training dataset.")
    parser.add_argument("--sample-rows", type=int, default=20, help="Number of rows for the checked-in example file.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    args = parser.parse_args()

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    full_df = generate_demo_dataset(n_rows=args.rows, seed=args.seed)
    raw_path = DATA_RAW_DIR / "workflows.csv"
    full_df.to_csv(raw_path, index=False)
    logger.info("Wrote %d synthetic rows to %s", len(full_df), raw_path)

    sample_df = generate_demo_dataset(n_rows=args.sample_rows, seed=args.seed + 1)
    sample_path = EXAMPLES_DIR / "sample_workflows.csv"
    sample_df.to_csv(sample_path, index=False)
    logger.info("Wrote %d sample rows to %s", len(sample_df), sample_path)


if __name__ == "__main__":
    main()
