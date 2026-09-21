from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from maritime_cbm.data.loader import (
    DatasetFormatError,
    compute_sha256,
    load_raw_dataset,
    split_features_targets,
)
from maritime_cbm.data.schema import (
    ALL_COLUMNS,
    EXPECTED_COLUMN_COUNT,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
)


def test_compute_sha256_reads_file_in_chunks(tmp_path: Path) -> None:
    content = b"maritime-cbm"
    path = tmp_path / "sample.bin"
    path.write_bytes(content)

    assert compute_sha256(path, chunk_size=3) == sha256(content).hexdigest()


def test_compute_sha256_rejects_non_positive_chunk_size(tmp_path: Path) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"content")

    with pytest.raises(ValueError, match="chunk_size"):
        compute_sha256(path, chunk_size=0)


def test_load_raw_dataset_applies_official_column_order(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    values = np.arange(EXPECTED_COLUMN_COUNT * 2, dtype=np.float64).reshape(
        2, EXPECTED_COLUMN_COUNT
    )
    np.savetxt(path, values)

    frame = load_raw_dataset(path)

    assert tuple(frame.columns) == ALL_COLUMNS
    assert frame.shape == (2, EXPECTED_COLUMN_COUNT)


def test_load_raw_dataset_rejects_wrong_column_count(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    np.savetxt(path, np.zeros((2, EXPECTED_COLUMN_COUNT - 1)))

    with pytest.raises(DatasetFormatError, match="Expected 18 columns"):
        load_raw_dataset(path)


def test_load_raw_dataset_rejects_non_numeric_values(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    row = ["invalid", *("0" for _ in range(EXPECTED_COLUMN_COUNT - 1))]
    path.write_text(" ".join(row), encoding="utf-8")

    with pytest.raises(DatasetFormatError, match="Failed to parse numeric data"):
        load_raw_dataset(path)


def test_split_features_targets_preserves_official_order(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    values = np.arange(EXPECTED_COLUMN_COUNT * 2, dtype=np.float64).reshape(
        2, EXPECTED_COLUMN_COUNT
    )
    np.savetxt(path, values)
    frame = load_raw_dataset(path)

    features, targets = split_features_targets(frame)

    assert tuple(features.columns) == FEATURE_COLUMNS
    assert tuple(targets.columns) == TARGET_COLUMNS
    features.iloc[0, 0] = -1
    assert frame.iloc[0, 0] != -1


def test_split_features_targets_rejects_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    np.savetxt(path, np.zeros((2, EXPECTED_COLUMN_COUNT)))
    frame = load_raw_dataset(path).drop(columns=["kMt"])

    with pytest.raises(DatasetFormatError, match="Required columns are missing"):
        split_features_targets(frame)
