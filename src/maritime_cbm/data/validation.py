"""Validation entry point for the official UCI naval propulsion release."""

import argparse
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import pandas as pd

from maritime_cbm.config import get_settings
from maritime_cbm.data.loader import DatasetFormatError, compute_sha256, load_raw_dataset
from maritime_cbm.data.schema import (
    ALL_COLUMNS,
    EXPECTED_COLUMN_COUNT,
    EXPECTED_GRID_SIZE,
    EXPECTED_KMC_VALUES,
    EXPECTED_KMT_VALUES,
    EXPECTED_ROW_COUNT,
    EXPECTED_SPEED_VALUES,
    GRID_ABSOLUTE_TOLERANCE,
    GRID_ROUND_DECIMALS,
    REQUIRED_RELEASE_FILES,
)
from maritime_cbm.logging_config import configure_logging

LOGGER = logging.getLogger(__name__)


class DatasetValidationError(ValueError):
    """Raised when the dataset violates a required invariant."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


@dataclass(frozen=True, slots=True)
class FileMetadata:
    """Reproducibility metadata for one release file."""

    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class DatasetObservations:
    """Non-failing observations that may affect later modeling decisions."""

    constant_columns: tuple[str, ...]
    duplicate_column_pairs: tuple[tuple[str, str], ...]
    lp_v_one_to_one: bool


@dataclass(frozen=True, slots=True)
class DatasetValidationReport:
    """Successful release validation result."""

    data_directory: str
    row_count: int
    column_count: int
    missing_value_count: int
    unique_speed_count: int
    unique_kmc_count: int
    unique_kmt_count: int
    files: tuple[FileMetadata, ...]
    observations: DatasetObservations

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the report."""
        return asdict(self)


def _grid_error(column: str, actual: np.ndarray, expected: Sequence[float]) -> str | None:
    expected_values = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected_values.shape or not np.allclose(
        actual,
        expected_values,
        rtol=0.0,
        atol=GRID_ABSOLUTE_TOLERANCE,
    ):
        return (
            f"{column} grid mismatch: expected {len(expected_values)} values "
            f"from {expected_values[0]} to {expected_values[-1]}, found {len(actual)}"
        )
    return None


def _duplicate_column_pairs(frame: pd.DataFrame) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for index, left in enumerate(frame.columns):
        for right in frame.columns[index + 1 :]:
            if frame[left].equals(frame[right]):
                pairs.append((str(left), str(right)))
    return tuple(pairs)


def _has_one_to_one_mapping(frame: pd.DataFrame, left: str, right: str) -> bool:
    unique_pairs = frame.loc[:, [left, right]].drop_duplicates()
    return bool(unique_pairs[left].is_unique and unique_pairs[right].is_unique)


def validate_dataframe(frame: pd.DataFrame) -> DatasetObservations:
    """Validate hard invariants and return non-failing feature observations."""
    errors: list[str] = []

    if frame.shape[0] != EXPECTED_ROW_COUNT:
        errors.append(f"Expected {EXPECTED_ROW_COUNT} rows, found {frame.shape[0]}")
    if frame.shape[1] != EXPECTED_COLUMN_COUNT:
        errors.append(f"Expected {EXPECTED_COLUMN_COUNT} columns, found {frame.shape[1]}")

    actual_columns = tuple(str(column) for column in frame.columns)
    if actual_columns != ALL_COLUMNS:
        errors.append(f"Column order mismatch: expected {ALL_COLUMNS}, found {actual_columns}")
        raise DatasetValidationError(errors)

    non_numeric_columns = tuple(
        column for column in ALL_COLUMNS if not pd.api.types.is_numeric_dtype(frame[column])
    )
    if non_numeric_columns:
        errors.append(f"Non-numeric columns found: {non_numeric_columns}")
    else:
        values = frame.to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(values).all():
            errors.append("Dataset contains missing or non-finite values")

    missing_value_count = int(frame.isna().sum().sum())
    if missing_value_count:
        errors.append(f"Dataset contains {missing_value_count} missing values")

    expected_grids: tuple[tuple[str, Sequence[float]], ...] = (
        ("v", EXPECTED_SPEED_VALUES),
        ("kMc", EXPECTED_KMC_VALUES),
        ("kMt", EXPECTED_KMT_VALUES),
    )
    for column, expected in expected_grids:
        actual = np.sort(frame[column].dropna().unique().astype(np.float64))
        if error := _grid_error(column, actual, expected):
            errors.append(error)

    normalized_grid = frame.loc[:, ["v", "kMc", "kMt"]].round(GRID_ROUND_DECIMALS)
    combination_counts = normalized_grid.value_counts(dropna=False)
    if len(combination_counts) != EXPECTED_GRID_SIZE or not bool(
        combination_counts.eq(1).all()
    ):
        errors.append(
            "Expected exactly one row for every rounded (v, kMc, kMt) grid combination"
        )

    if errors:
        raise DatasetValidationError(errors)

    constant_columns = tuple(
        str(column) for column in frame.columns if frame[column].nunique(dropna=False) == 1
    )
    return DatasetObservations(
        constant_columns=constant_columns,
        duplicate_column_pairs=_duplicate_column_pairs(frame),
        lp_v_one_to_one=_has_one_to_one_mapping(frame, "lp", "v"),
    )


def validate_release(data_directory: Path) -> DatasetValidationReport:
    """Validate required files and the complete official data grid."""
    missing_files = tuple(
        filename for filename in REQUIRED_RELEASE_FILES if not (data_directory / filename).is_file()
    )
    if missing_files:
        raise DatasetValidationError(
            [f"Missing required release files in {data_directory}: {missing_files}"]
        )

    file_metadata = tuple(
        FileMetadata(
            relative_path=path.name,
            size_bytes=path.stat().st_size,
            sha256=compute_sha256(path),
        )
        for path in (data_directory / filename for filename in REQUIRED_RELEASE_FILES)
    )
    empty_files = tuple(item.relative_path for item in file_metadata if item.size_bytes == 0)
    if empty_files:
        raise DatasetValidationError([f"Release files must not be empty: {empty_files}"])

    frame = load_raw_dataset(data_directory / "data.txt")
    observations = validate_dataframe(frame)
    return DatasetValidationReport(
        data_directory=str(data_directory),
        row_count=int(frame.shape[0]),
        column_count=int(frame.shape[1]),
        missing_value_count=int(frame.isna().sum().sum()),
        unique_speed_count=int(frame["v"].nunique()),
        unique_kmc_count=int(frame["kMc"].nunique()),
        unique_kmt_count=int(frame["kMt"].nunique()),
        files=file_metadata,
        observations=observations,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_directory",
        nargs="?",
        type=Path,
        default=get_settings().raw_data_dir,
        help="Directory containing data.txt, Features.txt, and README.txt",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run release validation and print a JSON report."""
    configure_logging()
    arguments = _build_parser().parse_args(argv)
    try:
        report = validate_release(arguments.data_directory)
    except (DatasetFormatError, DatasetValidationError, FileNotFoundError) as error:
        LOGGER.error("Dataset validation failed: %s", error)
        return 1

    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
