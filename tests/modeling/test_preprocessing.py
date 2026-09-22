import numpy as np
import pandas as pd
import pytest

from maritime_cbm.data.schema import MODEL_FEATURE_COLUMNS
from maritime_cbm.modeling.preprocessing import (
    SPEED_CENTERED_COLUMNS,
    ModelingInputError,
    SpeedConditionedCenterer,
)


def _feature_frame(speeds: list[float], offsets: list[float]) -> pd.DataFrame:
    rows = []
    for speed, offset in zip(speeds, offsets, strict=True):
        row = {column: speed * 100 + offset for column in MODEL_FEATURE_COLUMNS}
        row["v"] = speed
        rows.append(row)
    return pd.DataFrame(rows, columns=MODEL_FEATURE_COLUMNS)


def test_speed_centering_learns_train_only_means_and_retains_speed() -> None:
    train = _feature_frame([3.0, 3.0, 6.0, 6.0], [-1.0, 1.0, -2.0, 2.0])
    validation = _feature_frame([3.0, 6.0], [10.0, 20.0])

    transformer = SpeedConditionedCenterer().fit(train)
    centered_train = transformer.transform(train)
    centered_validation = transformer.transform(validation)

    assert centered_train.groupby("v")[list(SPEED_CENTERED_COLUMNS)].mean().to_numpy() == (
        pytest.approx(0.0)
    )
    assert centered_validation["v"].tolist() == [3.0, 6.0]
    assert centered_validation.loc[0, SPEED_CENTERED_COLUMNS[0]] == pytest.approx(10.0)
    assert centered_validation.loc[1, SPEED_CENTERED_COLUMNS[0]] == pytest.approx(20.0)
    assert tuple(centered_validation.columns) == MODEL_FEATURE_COLUMNS


def test_speed_centering_rejects_unseen_speed() -> None:
    transformer = SpeedConditionedCenterer().fit(_feature_frame([3.0, 6.0], [0.0, 0.0]))

    with pytest.raises(ModelingInputError, match="9.0"):
        transformer.transform(_feature_frame([9.0], [0.0]))


def test_speed_centering_rejects_missing_or_non_finite_features() -> None:
    frame = _feature_frame([3.0], [0.0])

    with pytest.raises(ModelingInputError, match="missing"):
        SpeedConditionedCenterer().fit(frame.drop(columns=["GTT"]))

    invalid = frame.copy()
    invalid.loc[0, "GTT"] = np.nan
    with pytest.raises(ModelingInputError, match="finite"):
        SpeedConditionedCenterer().fit(invalid)


def test_speed_centering_reports_canonical_feature_names() -> None:
    transformer = SpeedConditionedCenterer().fit(_feature_frame([3.0], [0.0]))

    assert tuple(transformer.get_feature_names_out()) == MODEL_FEATURE_COLUMNS
