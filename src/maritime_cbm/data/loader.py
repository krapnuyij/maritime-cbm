"""Raw UCI dataset loading and file hashing."""

from hashlib import sha256
from pathlib import Path

import pandas as pd

from maritime_cbm.data.schema import (
    ALL_COLUMNS,
    EXPECTED_COLUMN_COUNT,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
)


class DatasetFormatError(ValueError):
    """Raised when the raw dataset cannot be interpreted using the official schema."""


def compute_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest for a file without loading it fully into memory."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    digest = sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw_dataset(path: Path) -> pd.DataFrame:
    """Load the whitespace-delimited data file and apply the official column order."""
    if not path.is_file():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    try:
        frame = pd.read_csv(path, sep=r"\s+", header=None, dtype="float64")
    except (TypeError, ValueError, pd.errors.ParserError) as error:
        raise DatasetFormatError(f"Failed to parse numeric data from {path}") from error

    if frame.shape[1] != EXPECTED_COLUMN_COUNT:
        raise DatasetFormatError(
            f"Expected {EXPECTED_COLUMN_COUNT} columns, found {frame.shape[1]} in {path}"
        )

    frame.columns = list(ALL_COLUMNS)
    return frame


def split_features_targets(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return independent feature and target frames in the official column order."""
    missing_columns = tuple(column for column in ALL_COLUMNS if column not in frame.columns)
    if missing_columns:
        raise DatasetFormatError(f"Required columns are missing: {missing_columns}")

    features = frame.loc[:, FEATURE_COLUMNS].copy()
    targets = frame.loc[:, TARGET_COLUMNS].copy()
    return features, targets
