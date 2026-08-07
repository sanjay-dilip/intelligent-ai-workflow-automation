"""Reproducible preprocessing pipeline shared between training and inference.

Building the ColumnTransformer here, in one place, guarantees training and
prediction never preprocess data in different ways.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from workflow_ai.config import CATEGORICAL_FEATURE_COLUMNS, NUMERIC_FEATURE_COLUMNS


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Build the ColumnTransformer used identically at train and predict time.

    Numeric features are median-imputed and standard-scaled; categorical
    features are most-frequent-imputed and one-hot encoded, with unknown
    categories at inference time ignored rather than raising an error.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURE_COLUMNS)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURE_COLUMNS)),
        ]
    )
