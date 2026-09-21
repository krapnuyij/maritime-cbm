"""Shared feature selection for model training and inference."""

import pandas as pd

from maritime_cbm.data.schema import FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS


class FeatureSelectionError(ValueError):
    """Raised when raw model inputs do not contain the official feature schema."""


def select_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the 12 non-redundant model inputs in their canonical order."""
    missing_columns = tuple(column for column in FEATURE_COLUMNS if column not in frame.columns)
    if missing_columns:
        raise FeatureSelectionError(f"Required raw feature columns are missing: {missing_columns}")

    return frame.loc[:, MODEL_FEATURE_COLUMNS].copy()
