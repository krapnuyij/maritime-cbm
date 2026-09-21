from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from maritime_cbm.data.schema import (
    ALL_COLUMNS,
    EXPECTED_KMC_VALUES,
    EXPECTED_KMT_VALUES,
    EXPECTED_SPEED_VALUES,
)
from maritime_cbm.data.validation import (
    DatasetValidationError,
    main,
    validate_dataframe,
    validate_release,
)


@pytest.fixture(scope="module")
def valid_frame() -> pd.DataFrame:
    grid = np.asarray(
        list(product(EXPECTED_SPEED_VALUES, EXPECTED_KMC_VALUES, EXPECTED_KMT_VALUES)),
        dtype=np.float64,
    )
    row_index = np.arange(len(grid), dtype=np.float64)
    values = {
        column: row_index + column_index / 100 for column_index, column in enumerate(ALL_COLUMNS)
    }
    values["lp"] = grid[:, 0] / 3
    values["v"] = grid[:, 0]
    values["T1"] = np.full(len(grid), 288.0)
    values["P1"] = np.full(len(grid), 1.0)
    values["kMc"] = grid[:, 1]
    values["kMt"] = grid[:, 2]
    return pd.DataFrame(values, columns=ALL_COLUMNS)


def test_validate_dataframe_accepts_complete_grid(valid_frame: pd.DataFrame) -> None:
    observations = validate_dataframe(valid_frame)

    assert observations.constant_columns == ("T1", "P1")
    assert observations.duplicate_column_pairs == ()
    assert observations.lp_v_one_to_one is True


def test_validate_dataframe_rejects_missing_row(valid_frame: pd.DataFrame) -> None:
    with pytest.raises(DatasetValidationError, match="Expected 11934 rows"):
        validate_dataframe(valid_frame.iloc[:-1])


def test_validate_dataframe_rejects_grid_mismatch(valid_frame: pd.DataFrame) -> None:
    invalid_frame = valid_frame.copy()
    invalid_frame.loc[0, "v"] = 4.0

    with pytest.raises(DatasetValidationError, match="v grid mismatch"):
        validate_dataframe(invalid_frame)


def test_validate_dataframe_accepts_grid_within_tolerance(valid_frame: pd.DataFrame) -> None:
    perturbed_frame = valid_frame.copy()
    perturbed_frame.loc[perturbed_frame["v"] == 3.0, "v"] += 1e-7

    validate_dataframe(perturbed_frame)


def test_validate_dataframe_rejects_missing_value(valid_frame: pd.DataFrame) -> None:
    invalid_frame = valid_frame.copy()
    invalid_frame.loc[0, "GTT"] = np.nan

    with pytest.raises(DatasetValidationError, match="missing or non-finite"):
        validate_dataframe(invalid_frame)


def test_validate_dataframe_rejects_duplicate_grid_combination(
    valid_frame: pd.DataFrame,
) -> None:
    invalid_frame = valid_frame.copy()
    invalid_frame.loc[1, ["v", "kMc", "kMt"]] = invalid_frame.loc[0, ["v", "kMc", "kMt"]].to_numpy()

    with pytest.raises(DatasetValidationError, match="exactly one row"):
        validate_dataframe(invalid_frame)


def test_validate_dataframe_observes_duplicate_columns(valid_frame: pd.DataFrame) -> None:
    duplicate_frame = valid_frame.copy()
    duplicate_frame["Tp"] = duplicate_frame["Ts"]

    observations = validate_dataframe(duplicate_frame)

    assert ("Ts", "Tp") in observations.duplicate_column_pairs


def test_validate_dataframe_observes_non_unique_lp_v_mapping(
    valid_frame: pd.DataFrame,
) -> None:
    non_unique_frame = valid_frame.copy()
    first_lp = float(non_unique_frame.loc[non_unique_frame["v"] == 3.0, "lp"].iloc[0])
    non_unique_frame.loc[non_unique_frame["v"] == 6.0, "lp"] = first_lp

    observations = validate_dataframe(non_unique_frame)

    assert observations.lp_v_one_to_one is False


def test_validate_dataframe_rejects_column_order(valid_frame: pd.DataFrame) -> None:
    reordered_columns = [*ALL_COLUMNS[1:], ALL_COLUMNS[0]]

    with pytest.raises(DatasetValidationError, match="Column order mismatch"):
        validate_dataframe(valid_frame.loc[:, reordered_columns])


def test_validate_release_reports_file_metadata(tmp_path: Path, valid_frame: pd.DataFrame) -> None:
    np.savetxt(tmp_path / "data.txt", valid_frame.to_numpy())
    (tmp_path / "Features.txt").write_text("feature descriptions", encoding="utf-8")
    (tmp_path / "README.txt").write_text("release documentation", encoding="utf-8")

    report = validate_release(tmp_path)

    assert report.row_count == len(valid_frame)
    assert report.column_count == len(ALL_COLUMNS)
    assert {item.relative_path for item in report.files} == {
        "data.txt",
        "Features.txt",
        "README.txt",
    }
    assert all(len(item.sha256) == 64 for item in report.files)


def test_validation_main_returns_failure_for_missing_release(tmp_path: Path) -> None:
    assert main([str(tmp_path)]) == 1
