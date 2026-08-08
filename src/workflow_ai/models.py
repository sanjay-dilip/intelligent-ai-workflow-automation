"""Unfit model pipeline builders for workflow risk classification.

Each builder returns a fresh, unfit scikit-learn Pipeline pairing the
shared preprocessing step with a classifier. No fitting or I/O happens
here -- that is train.py's job. Every call builds its own preprocessing
ColumnTransformer instance so pipelines never share fitted state.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from workflow_ai.config import RANDOM_SEED
from workflow_ai.preprocessing import build_preprocessing_pipeline

RANDOM_FOREST_N_ESTIMATORS: int = 200
LOGISTIC_REGRESSION_MAX_ITER: int = 1000


def build_random_forest_pipeline() -> Pipeline:
    """Build the primary model: preprocessing + a class-balanced Random Forest."""
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessing_pipeline()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=RANDOM_FOREST_N_ESTIMATORS,
                    random_state=RANDOM_SEED,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def build_logistic_regression_baseline() -> Pipeline:
    """Build a Logistic Regression baseline pipeline for comparison, never persisted.

    max_iter is raised from scikit-learn's default of 100 because the
    standardized + one-hot-encoded feature space can otherwise fail to
    converge; do not lower this back to the default.
    """
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessing_pipeline()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=LOGISTIC_REGRESSION_MAX_ITER,
                    random_state=RANDOM_SEED,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def build_dummy_baseline(strategy: str = "stratified") -> Pipeline:
    """Build a Dummy baseline pipeline for comparison, never persisted.

    The "stratified" strategy samples predictions from the training
    class distribution, giving a more informative macro-F1 floor than
    always predicting the plurality class.
    """
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessing_pipeline()),
            ("classifier", DummyClassifier(strategy=strategy, random_state=RANDOM_SEED)),
        ]
    )


PRIMARY_MODEL_NAME: str = "random_forest"

BASELINE_BUILDERS = {
    "logistic_regression": build_logistic_regression_baseline,
    "dummy": build_dummy_baseline,
}
