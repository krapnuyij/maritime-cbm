import pandas as pd
import pytest

from maritime_cbm.data.features import FeatureSelectionError, select_model_features
from maritime_cbm.data.schema import (
    ALL_COLUMNS,
    DROPPED_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMNS,
)


def test_select_model_features_uses_canonical_non_redundant_columns() -> None:
    frame = pd.DataFrame([{column: index for index, column in enumerate(FEATURE_COLUMNS)}])

    selected = select_model_features(frame)

    assert tuple(selected.columns) == MODEL_FEATURE_COLUMNS
    assert len(selected.columns) == 12
    assert set(selected.columns).isdisjoint(DROPPED_FEATURE_COLUMNS)


def test_select_model_features_returns_independent_frame() -> None:
    frame = pd.DataFrame([{column: index for index, column in enumerate(FEATURE_COLUMNS)}])

    selected = select_model_features(frame)
    selected.iloc[0, 0] = -1

    assert frame.loc[0, MODEL_FEATURE_COLUMNS[0]] != -1


def test_select_model_features_excludes_targets_from_full_dataset_frame() -> None:
    frame = pd.DataFrame([{column: index for index, column in enumerate(ALL_COLUMNS)}])

    selected = select_model_features(frame)

    assert tuple(selected.columns) == MODEL_FEATURE_COLUMNS
    assert set(selected.columns).isdisjoint(TARGET_COLUMNS)


def test_select_model_features_rejects_missing_raw_column() -> None:
    frame = pd.DataFrame([{column: 0.0 for column in FEATURE_COLUMNS if column != "GTT"}])

    with pytest.raises(FeatureSelectionError, match="GTT"):
        select_model_features(frame)
