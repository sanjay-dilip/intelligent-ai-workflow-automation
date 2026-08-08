"""Data validation layer for workflow records.

Validation reports errors and warnings; it never silently repairs invalid
business data. Errors indicate the dataset should not be used for
downstream processing as-is. Warnings flag things worth a human's
attention (e.g. an unrecognized status category) without blocking the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from workflow_ai.config import (
    CRITICAL_COLUMNS,
    DATETIME_COLUMNS,
    NON_NEGATIVE_COLUMNS,
    REQUIRED_COLUMNS,
    UNIT_INTERVAL_COLUMNS,
    VALID_STATUSES,
    WORKFLOW_ID_COLUMN,
)

_ERROR = "error"
_WARNING = "warning"


@dataclass
class ValidationIssue:
    """A single validation finding."""

    severity: str
    check: str
    message: str
    workflow_ids: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class ValidationResult:
    """Aggregate result of validating a workflow dataset."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        """Issues severe enough to block downstream processing."""
        return [issue for issue in self.issues if issue.severity == _ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Non-blocking issues worth reviewing."""
        return [issue for issue in self.issues if issue.severity == _WARNING]

    @property
    def is_valid(self) -> bool:
        """True if there are no blocking errors."""
        return len(self.errors) == 0

    def summary(self) -> str:
        """A human-readable, one-line-per-issue summary."""
        if not self.issues:
            return "No validation issues found."
        return "\n".join(
            f"{issue.severity.upper()} [{issue.check}]: {issue.message}" for issue in self.issues
        )


def validate_workflow_data(df: pd.DataFrame) -> ValidationResult:
    """Validate a workflow dataset and return all discovered issues.

    A missing-required-columns error short-circuits the remaining
    column-level checks, since those checks would otherwise raise
    ``KeyError`` on a column that was never present.
    """
    result = ValidationResult()

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        result.issues.append(
            ValidationIssue(
                severity=_ERROR,
                check="required_columns",
                message=f"Missing required columns: {missing_columns}",
            )
        )
        return result

    _check_duplicate_ids(df, result)
    _check_nulls_in_critical_fields(df, result)
    _check_non_negative_columns(df, result)
    _check_unit_interval_columns(df, result)
    _check_sla_validity(df, result)
    _check_timestamp_consistency(df, result)
    _check_unknown_status(df, result)

    return result


def _workflow_ids_for(df: pd.DataFrame, mask: pd.Series) -> tuple[str, ...]:
    if WORKFLOW_ID_COLUMN not in df.columns:
        return tuple()
    return tuple(df.loc[mask, WORKFLOW_ID_COLUMN].astype(str))


def _check_duplicate_ids(df: pd.DataFrame, result: ValidationResult) -> None:
    duplicate_mask = df[WORKFLOW_ID_COLUMN].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_ids = tuple(sorted(set(df.loc[duplicate_mask, WORKFLOW_ID_COLUMN].astype(str))))
        result.issues.append(
            ValidationIssue(
                severity=_ERROR,
                check="duplicate_workflow_id",
                message=f"Duplicate workflow_id values found: {list(duplicate_ids)}",
                workflow_ids=duplicate_ids,
            )
        )


def _check_nulls_in_critical_fields(df: pd.DataFrame, result: ValidationResult) -> None:
    for column in CRITICAL_COLUMNS:
        if column not in df.columns:
            continue
        series = df[column]
        if column in DATETIME_COLUMNS:
            series = pd.to_datetime(series, errors="coerce")
        null_mask = series.isna()
        if null_mask.any():
            result.issues.append(
                ValidationIssue(
                    severity=_ERROR,
                    check="null_critical_field",
                    message=f"Column '{column}' has {int(null_mask.sum())} null value(s)",
                    workflow_ids=_workflow_ids_for(df, null_mask),
                )
            )


def _check_non_negative_columns(df: pd.DataFrame, result: ValidationResult) -> None:
    for column in NON_NEGATIVE_COLUMNS:
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        negative_mask = numeric < 0
        if negative_mask.any():
            result.issues.append(
                ValidationIssue(
                    severity=_ERROR,
                    check="negative_value",
                    message=f"Column '{column}' has {int(negative_mask.sum())} negative value(s)",
                    workflow_ids=_workflow_ids_for(df, negative_mask),
                )
            )


def _check_unit_interval_columns(df: pd.DataFrame, result: ValidationResult) -> None:
    for column in UNIT_INTERVAL_COLUMNS:
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        out_of_bounds_mask = (numeric < 0) | (numeric > 1)
        if out_of_bounds_mask.any():
            over_one_count = int((numeric > 1).sum())
            message = (
                f"Column '{column}' has {int(out_of_bounds_mask.sum())} value(s) outside [0, 1]"
            )
            if over_one_count > 0:
                message += (
                    f" ({over_one_count} value(s) exceed 1 -- check whether the source uses a"
                    " 0-100 percentage scale instead of the expected 0-1 fraction)"
                )
            result.issues.append(
                ValidationIssue(
                    severity=_ERROR,
                    check="unit_interval_bounds",
                    message=message,
                    workflow_ids=_workflow_ids_for(df, out_of_bounds_mask),
                )
            )


def _check_sla_validity(df: pd.DataFrame, result: ValidationResult) -> None:
    numeric = pd.to_numeric(df["sla_hours"], errors="coerce")
    invalid_mask = numeric <= 0
    if invalid_mask.any():
        result.issues.append(
            ValidationIssue(
                severity=_ERROR,
                check="impossible_sla",
                message=f"Column 'sla_hours' has {int(invalid_mask.sum())} value(s) <= 0",
                workflow_ids=_workflow_ids_for(df, invalid_mask),
            )
        )


def _check_timestamp_consistency(df: pd.DataFrame, result: ValidationResult) -> None:
    created = pd.to_datetime(df["created_at"], errors="coerce")
    updated = pd.to_datetime(df["updated_at"], errors="coerce")
    inconsistent_mask = updated < created
    if inconsistent_mask.any():
        result.issues.append(
            ValidationIssue(
                severity=_ERROR,
                check="inconsistent_timestamps",
                message=f"{int(inconsistent_mask.sum())} row(s) have updated_at before created_at",
                workflow_ids=_workflow_ids_for(df, inconsistent_mask),
            )
        )


def _check_unknown_status(df: pd.DataFrame, result: ValidationResult) -> None:
    unknown_mask = ~df["status"].isin(VALID_STATUSES)
    if unknown_mask.any():
        unknown_values = tuple(sorted(set(df.loc[unknown_mask, "status"].astype(str))))
        result.issues.append(
            ValidationIssue(
                severity=_WARNING,
                check="unknown_status",
                message=f"Unrecognized status value(s): {list(unknown_values)}",
                workflow_ids=_workflow_ids_for(df, unknown_mask),
            )
        )
