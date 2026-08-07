"""Load workflow data from CSV files.

Loading only parses column types (e.g. timestamps); it never repairs,
drops, or otherwise silently alters invalid business values. Data quality
issues are surfaced separately by the validation layer.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from workflow_ai.config import DATETIME_COLUMNS

logger = logging.getLogger(__name__)


def load_workflow_data(path: Path) -> pd.DataFrame:
    """Load a workflow CSV file into a DataFrame, parsing timestamp columns.

    Args:
        path: Path to a CSV file containing workflow records.

    Returns:
        The loaded DataFrame. Timestamp columns are parsed to datetime;
        values that fail to parse become NaT so validation can flag them
        rather than the load step guessing at a repair.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Workflow data file not found: {path}")

    logger.info("Loading workflow data from %s", path)
    df = pd.read_csv(path)

    for column in DATETIME_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    logger.info("Loaded %d rows from %s", len(df), path)
    return df
