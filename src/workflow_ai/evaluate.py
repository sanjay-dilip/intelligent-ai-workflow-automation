"""Evaluation of fitted classification pipelines.

All metrics here are computed on the synthetic demonstration dataset
(scripts/generate_demo_data.py). They are useful for comparing the
primary model against baselines on this project's synthetic data, but
they do not indicate real-world predictive performance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from workflow_ai.config import RISK_LABELS


@dataclass(frozen=True)
class EvaluationResult:
    """Structured classification metrics for one fitted pipeline on one test set."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    confusion_matrix: list[list[int]]
    labels: list[str]
    support: dict[str, int]

    def to_dict(self) -> dict:
        """Return a plain, JSON-serializable dict of this result."""
        return asdict(self)


def evaluate_pipeline(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> EvaluationResult:
    """Score a fitted pipeline against a held-out test set.

    Labels are always evaluated in RISK_LABELS order, and zero_division
    is treated as 0 rather than raising or warning, since a small
    synthetic test split can plausibly contain zero predictions for a
    given class (especially for the Dummy baseline).
    """
    labels = list(RISK_LABELS)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, average="weighted", zero_division=0
    )
    matrix = confusion_matrix(y_test, y_pred, labels=labels).tolist()
    support = y_test.value_counts().reindex(labels, fill_value=0).astype(int).to_dict()

    return EvaluationResult(
        accuracy=float(accuracy),
        precision_macro=float(precision_macro),
        recall_macro=float(recall_macro),
        f1_macro=float(f1_macro),
        precision_weighted=float(precision_weighted),
        recall_weighted=float(recall_weighted),
        f1_weighted=float(f1_weighted),
        confusion_matrix=matrix,
        labels=labels,
        support=support,
    )


def format_evaluation_summary(result: EvaluationResult) -> str:
    """Render an EvaluationResult as a short, human-readable summary line."""
    return (
        f"accuracy={result.accuracy:.3f} "
        f"f1_macro={result.f1_macro:.3f} f1_weighted={result.f1_weighted:.3f} "
        f"support={result.support}"
    )
