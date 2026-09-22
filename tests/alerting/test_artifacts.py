from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from maritime_cbm.alerting.evaluation import (
    AlertArtifactError,
    _verify_hash,
    _verify_manifest_common,
    validate_prediction_table,
)
from maritime_cbm.data.loader import compute_sha256
from maritime_cbm.data.splitting import DatasetSplit


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "v": [3.0, 6.0, 9.0, 3.0],
            "kMc": [0.95, 0.96, 0.97, 0.98],
            "kMt": [0.975, 0.980, 0.985, 0.990],
        }
    )


def _splits() -> dict[str, DatasetSplit]:
    indices = np.asarray([0, 1], dtype=np.int64)
    return {
        scenario: DatasetSplit(
            name=scenario,
            train=np.asarray([2], dtype=np.int64),
            validation=np.asarray([3], dtype=np.int64),
            test=indices,
        )
        for scenario in (
            "random_row",
            "state_group",
            "compressor_holdout",
            "turbine_holdout",
        )
    }


def _predictions() -> pd.DataFrame:
    rows = []
    frame = _frame()
    for scenario, split in _splits().items():
        for row_index in split.test:
            row = frame.iloc[row_index]
            rows.append(
                {
                    "scenario": scenario,
                    "role": "test",
                    "row_index": row_index,
                    "v": row["v"],
                    "kMc_true": row["kMc"],
                    "kMc_predicted": row["kMc"] + 0.001,
                    "kMc_error": 0.001,
                    "kMt_true": row["kMt"],
                    "kMt_predicted": row["kMt"] - 0.001,
                    "kMt_error": -0.001,
                }
            )
    return pd.DataFrame(rows)


def test_prediction_artifact_matches_current_test_rows() -> None:
    validate_prediction_table(_frame(), _predictions(), _splits())


def test_prediction_artifact_rejects_changed_truth() -> None:
    predictions = _predictions()
    predictions.loc[0, "kMc_true"] = 0.999

    with pytest.raises(AlertArtifactError, match="truths differ"):
        validate_prediction_table(_frame(), predictions, _splits())


def test_prediction_artifact_rejects_duplicate_rows() -> None:
    predictions = pd.concat([_predictions(), _predictions().iloc[:1]], ignore_index=True)

    with pytest.raises(AlertArtifactError, match="duplicate"):
        validate_prediction_table(_frame(), predictions, _splits())


def test_prediction_artifact_rejects_fractional_row_index() -> None:
    predictions = _predictions()
    predictions["row_index"] = predictions["row_index"].astype(np.float64)
    predictions.loc[0, "row_index"] = 0.5

    with pytest.raises(AlertArtifactError, match="must be integers"):
        validate_prediction_table(_frame(), predictions, _splits())


def test_upstream_artifact_rejects_sha256_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"trusted")

    assert _verify_hash(artifact, compute_sha256(artifact), label="test") == compute_sha256(
        artifact
    )
    with pytest.raises(AlertArtifactError, match="SHA-256 differs"):
        _verify_hash(artifact, "0" * 64, label="test")


def test_upstream_manifest_rejects_split_or_runtime_mismatch() -> None:
    split_hashes = {"state_group": {"train": "train", "test": "test"}}
    versions = {"python": "3.13.15"}
    payload: dict[str, object] = {
        "split_hashes": split_hashes,
        "runtime_versions": versions,
    }

    _verify_manifest_common(
        payload,
        split_hashes=split_hashes,
        expected_versions=versions,
        label="test",
    )
    with pytest.raises(AlertArtifactError, match="split hashes differ"):
        _verify_manifest_common(
            {**payload, "split_hashes": {}},
            split_hashes=split_hashes,
            expected_versions=versions,
            label="test",
        )
    with pytest.raises(AlertArtifactError, match="runtime versions differ"):
        _verify_manifest_common(
            {**payload, "runtime_versions": {}},
            split_hashes=split_hashes,
            expected_versions=versions,
            label="test",
        )
