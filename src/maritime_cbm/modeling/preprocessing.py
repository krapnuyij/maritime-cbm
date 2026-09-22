"""Leakage-safe feature preprocessing for baseline models."""

from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from maritime_cbm.data.schema import MODEL_FEATURE_COLUMNS

SPEED_COLUMN = "v"
SPEED_CENTERED_COLUMNS: tuple[str, ...] = tuple(
    column for column in MODEL_FEATURE_COLUMNS if column != SPEED_COLUMN
)


class ModelingInputError(ValueError):
    """Raised when model input cannot satisfy the approved feature contract."""


def _canonical_feature_frame(values: object) -> pd.DataFrame:
    if not isinstance(values, pd.DataFrame):
        raise ModelingInputError("Model features must be provided as a pandas DataFrame")

    missing_columns = tuple(
        column for column in MODEL_FEATURE_COLUMNS if column not in values.columns
    )
    if missing_columns:
        raise ModelingInputError(f"Required model columns are missing: {missing_columns}")

    frame = values.loc[:, MODEL_FEATURE_COLUMNS].copy()
    try:
        numeric_values = frame.to_numpy(dtype=np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise ModelingInputError("Model features must be numeric") from error
    if not np.isfinite(numeric_values).all():
        raise ModelingInputError("Model features must contain only finite values")
    return frame


class SpeedConditionedCenterer(TransformerMixin, BaseEstimator):
    """Center eleven sensors by train-only speed means while retaining ship speed."""

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Learn per-speed sensor means from the training feature frame."""
        del y
        frame = _canonical_feature_frame(X)
        if frame.empty:
            raise ModelingInputError("Cannot fit speed centering on an empty frame")

        self.speed_means_ = frame.groupby(SPEED_COLUMN, sort=True)[
            list(SPEED_CENTERED_COLUMNS)
        ].mean()
        self.known_speeds_ = self.speed_means_.index.to_numpy(dtype=np.float64, copy=True)
        self.feature_names_in_ = np.asarray(MODEL_FEATURE_COLUMNS, dtype=object)
        self.n_features_in_ = len(MODEL_FEATURE_COLUMNS)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply train-only per-speed centering and preserve canonical column order."""
        check_is_fitted(self, ("speed_means_", "known_speeds_"))
        frame = _canonical_feature_frame(X)
        speeds = frame[SPEED_COLUMN].to_numpy(dtype=np.float64, copy=False)
        unknown_speeds = np.setdiff1d(np.unique(speeds), self.known_speeds_)
        if len(unknown_speeds):
            raise ModelingInputError(
                f"No training statistics are available for speeds: {unknown_speeds.tolist()}"
            )

        centered = frame.copy()
        centers = self.speed_means_.loc[speeds, list(SPEED_CENTERED_COLUMNS)].to_numpy()
        centered.loc[:, list(SPEED_CENTERED_COLUMNS)] = (
            frame.loc[:, list(SPEED_CENTERED_COLUMNS)].to_numpy() - centers
        )
        return centered

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return the unchanged canonical feature names."""
        del input_features
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_.copy()
