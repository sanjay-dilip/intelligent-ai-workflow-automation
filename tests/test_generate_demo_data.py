"""Tests for the synthetic demo data generator."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_demo_data import generate_demo_dataset  # noqa: E402
from workflow_ai.config import REQUIRED_COLUMNS, RISK_LABELS, TARGET_COLUMN  # noqa: E402


def test_generated_dataset_has_required_schema() -> None:
    df = generate_demo_dataset(n_rows=200, seed=1)
    for column in REQUIRED_COLUMNS:
        assert column in df.columns
    assert TARGET_COLUMN in df.columns


def test_generated_risk_labels_are_valid_categories() -> None:
    df = generate_demo_dataset(n_rows=200, seed=1)
    assert set(df[TARGET_COLUMN].unique()) <= set(RISK_LABELS)


def test_generation_is_deterministic_for_fixed_seed() -> None:
    first = generate_demo_dataset(n_rows=200, seed=7)
    second = generate_demo_dataset(n_rows=200, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_produce_different_data() -> None:
    first = generate_demo_dataset(n_rows=200, seed=1)
    second = generate_demo_dataset(n_rows=200, seed=2)
    assert not first[TARGET_COLUMN].equals(second[TARGET_COLUMN])


def test_risk_labels_are_not_perfectly_separable_by_class_presence() -> None:
    """All three classes should appear with a reasonably large sample -- a
    label scheme that collapses to one or two classes would indicate the
    noise/overlap in the soft-label sampling is not working as intended."""
    df = generate_demo_dataset(n_rows=2000, seed=42)
    counts = df[TARGET_COLUMN].value_counts()
    assert set(counts.index) == set(RISK_LABELS)
    assert counts.min() > 50


def test_no_negative_or_out_of_bounds_values() -> None:
    df = generate_demo_dataset(n_rows=500, seed=3)
    assert (df["sla_hours"] > 0).all()
    assert (df["elapsed_hours"] >= 0).all()
    assert (df["error_count"] >= 0).all()
    assert (df["rework_count"] >= 0).all()
    assert (df["transaction_volume"] >= 1).all()
    assert df["completion_rate"].between(0, 1).all()
    assert df["automation_score"].between(0, 1).all()
