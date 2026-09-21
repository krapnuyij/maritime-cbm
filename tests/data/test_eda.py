from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from maritime_cbm.data.eda import (
    CORRELATION_FILENAME,
    SENSOR_FEATURE_COLUMNS,
    SPEED_CORRELATION_FILENAME,
    SPEED_VARIANCE_FILENAME,
    SUMMARY_FILENAME,
    build_between_speed_variance_ratios,
    build_speed_conditioned_correlations,
    build_summary_statistics,
    build_train_correlation_matrix,
    write_eda_figures,
    write_eda_tables,
)
from maritime_cbm.data.schema import ALL_COLUMNS, MODEL_FEATURE_COLUMNS, TARGET_COLUMNS
from maritime_cbm.data.splitting import DatasetSplit


@pytest.fixture
def analysis_frame() -> pd.DataFrame:
    row_count = 12
    frame = pd.DataFrame(
        {
            column: np.arange(row_count, dtype=np.float64) + column_index
            for column_index, column in enumerate(ALL_COLUMNS)
        }
    )
    frame["v"] = np.repeat([3.0, 6.0], row_count // 2)
    return frame


@pytest.fixture
def small_split() -> DatasetSplit:
    return DatasetSplit(
        name="state_group",
        train=np.arange(8, dtype=np.int64),
        validation=np.arange(8, 10, dtype=np.int64),
        test=np.arange(10, 12, dtype=np.int64),
    )


def test_eda_tables_have_canonical_scope(
    analysis_frame: pd.DataFrame,
    small_split: DatasetSplit,
) -> None:
    summary = build_summary_statistics(analysis_frame)
    correlation = build_train_correlation_matrix(analysis_frame, small_split)

    assert tuple(summary["variable"]) == ALL_COLUMNS
    assert tuple(correlation["variable"]) == MODEL_FEATURE_COLUMNS + TARGET_COLUMNS
    assert tuple(correlation.columns[1:]) == MODEL_FEATURE_COLUMNS + TARGET_COLUMNS


def test_speed_conditioned_tables_use_train_rows_and_sensor_features(
    analysis_frame: pd.DataFrame,
    small_split: DatasetSplit,
) -> None:
    correlations = build_speed_conditioned_correlations(analysis_frame, small_split)
    variance_ratios = build_between_speed_variance_ratios(analysis_frame, small_split)

    assert set(correlations["speed_knots"]) == {3.0, 6.0}
    assert tuple(correlations["feature"].drop_duplicates()) == SENSOR_FEATURE_COLUMNS
    assert tuple(correlations["target"].drop_duplicates()) == TARGET_COLUMNS
    assert tuple(variance_ratios["feature"]) == SENSOR_FEATURE_COLUMNS
    assert variance_ratios["between_speed_variance_ratio"].between(0.0, 1.0).all()


def test_write_eda_tables_does_not_require_plotting_dependency(
    tmp_path: Path,
    analysis_frame: pd.DataFrame,
    small_split: DatasetSplit,
) -> None:
    paths = write_eda_tables(analysis_frame, small_split, tmp_path)

    assert {path.name for path in paths} == {
        SUMMARY_FILENAME,
        CORRELATION_FILENAME,
        SPEED_CORRELATION_FILENAME,
        SPEED_VARIANCE_FILENAME,
    }
    assert all(path.is_file() for path in paths)


def test_write_eda_figures_creates_three_png_files(
    tmp_path: Path,
    analysis_frame: pd.DataFrame,
    small_split: DatasetSplit,
) -> None:
    pytest.importorskip("matplotlib")

    paths = write_eda_figures(analysis_frame, small_split, tmp_path)

    assert len(paths) == 3
    assert all(path.suffix == ".png" and path.stat().st_size > 0 for path in paths)
