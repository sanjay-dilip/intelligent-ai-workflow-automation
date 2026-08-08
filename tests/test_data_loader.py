"""Tests for the workflow data loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from workflow_ai.data_loader import load_workflow_data


def test_load_workflow_data_parses_timestamps(tmp_path: Path) -> None:
    csv_path = tmp_path / "workflows.csv"
    pd.DataFrame(
        {
            "workflow_id": ["WF-1"],
            "created_at": ["2026-01-01"],
            "updated_at": ["2026-01-02"],
        }
    ).to_csv(csv_path, index=False)

    df = load_workflow_data(csv_path)

    assert pd.api.types.is_datetime64_any_dtype(df["created_at"])
    assert pd.api.types.is_datetime64_any_dtype(df["updated_at"])


def test_load_workflow_data_coerces_unparsable_timestamp_to_nat(tmp_path: Path) -> None:
    csv_path = tmp_path / "workflows.csv"
    pd.DataFrame({"workflow_id": ["WF-1"], "created_at": ["not-a-date"]}).to_csv(
        csv_path, index=False
    )

    df = load_workflow_data(csv_path)

    assert pd.isna(df.loc[0, "created_at"])


def test_load_workflow_data_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_workflow_data(tmp_path / "does_not_exist.csv")


def test_load_workflow_data_row_count(tmp_path: Path) -> None:
    csv_path = tmp_path / "workflows.csv"
    pd.DataFrame({"workflow_id": ["WF-1", "WF-2", "WF-3"]}).to_csv(csv_path, index=False)

    df = load_workflow_data(csv_path)

    assert len(df) == 3
