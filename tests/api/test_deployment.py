import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from maritime_cbm.api.deployment import (
    DEPLOYMENT_CANDIDATE_ID,
    DEPLOYMENT_MODEL_VERSION,
    DEPLOYMENT_POLICY_VERSION,
    DeploymentContractError,
    load_deployment,
    load_deployment_contract,
    torch_base_version,
)
from maritime_cbm.config import DEFAULT_DEPLOYMENT_CONTRACT_PATH, DEFAULT_RAW_DATA_DIR
from maritime_cbm.data.features import select_model_features
from maritime_cbm.data.loader import compute_sha256, load_raw_dataset
from maritime_cbm.data.schema import MODEL_FEATURE_COLUMNS, TARGET_COLUMNS
from maritime_cbm.data.splitting import build_dataset_splits
from maritime_cbm.modeling.preprocessing import SPEED_CENTERED_COLUMNS
from maritime_cbm.modeling.torch_models import get_torch_candidate_spec
from maritime_cbm.modeling.torch_training import (
    TrainingConfig,
    fit_torch_regressor,
    save_torch_checkpoint,
)


def _synthetic_training_data() -> tuple[pd.DataFrame, np.ndarray]:
    row_count = 27
    speeds = np.tile(np.asarray([3.0, 6.0, 9.0]), 9)
    values = np.arange(row_count * len(MODEL_FEATURE_COLUMNS), dtype=np.float64).reshape(
        row_count, len(MODEL_FEATURE_COLUMNS)
    )
    frame = pd.DataFrame(values / 100.0 + 1.0, columns=MODEL_FEATURE_COLUMNS)
    frame.loc[:, "v"] = speeds
    targets = np.column_stack(
        (
            0.975 + frame["T2"].to_numpy() * 1e-4,
            0.9875 + frame["P2"].to_numpy() * 5e-5,
        )
    )
    return frame, targets


def _write_synthetic_bundle(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    features, targets = _synthetic_training_data()
    result = fit_torch_regressor(
        features.iloc[:18],
        targets[:18],
        features.iloc[18:],
        targets[18:],
        candidate=get_torch_candidate_spec(DEPLOYMENT_CANDIDATE_ID),
        seed=42,
        config=TrainingConfig(
            batch_size=9,
            max_epochs=1,
            min_epochs=1,
            patience=1,
        ),
    )
    checkpoint_path = tmp_path / "model.pt"
    save_torch_checkpoint(
        result,
        checkpoint_path,
        metadata={"scenario": "state_group", "role": "selection_validation", "seed": 42},
    )
    contract: dict[str, object] = {
        "schema_version": 1,
        "model_version": DEPLOYMENT_MODEL_VERSION,
        "candidate_id": DEPLOYMENT_CANDIDATE_ID,
        "checkpoint_sha256": compute_sha256(checkpoint_path),
        "checkpoint_seed": 42,
        "checkpoint_scenario": "state_group",
        "torch_base_version": torch_base_version(torch.__version__),
        "input_features": list(MODEL_FEATURE_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "allowed_speeds": [3.0, 6.0, 9.0],
        "continuous_feature_bounds": {
            feature: {"minimum": 0.0, "maximum": 100.0} for feature in SPEED_CENTERED_COLUMNS
        },
        "policy_version": DEPLOYMENT_POLICY_VERSION,
        "watch_severity_threshold": 0.5,
        "alert_severity_threshold": 0.8,
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    return contract_path, checkpoint_path, contract


def test_load_deployment_validates_synthetic_contract_and_checkpoint(tmp_path: Path) -> None:
    contract_path, checkpoint_path, _ = _write_synthetic_bundle(tmp_path)

    deployment = load_deployment(contract_path, checkpoint_path)

    assert deployment.contract.model_version == DEPLOYMENT_MODEL_VERSION
    assert deployment.checkpoint.seed == 42
    assert deployment.checkpoint.regressor.preprocessor.speed_values.tolist() == [3.0, 6.0, 9.0]


def test_load_deployment_rejects_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    contract_path, checkpoint_path, contract = _write_synthetic_bundle(tmp_path)
    contract["checkpoint_sha256"] = "0" * 64
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(DeploymentContractError, match="SHA-256"):
        load_deployment(contract_path, checkpoint_path)


def test_load_deployment_rejects_torch_base_version_mismatch(tmp_path: Path) -> None:
    contract_path, checkpoint_path, contract = _write_synthetic_bundle(tmp_path)
    contract["torch_base_version"] = "0.0.0"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(DeploymentContractError, match="PyTorch base versions"):
        load_deployment(contract_path, checkpoint_path)


def test_contract_rejects_feature_order_drift(tmp_path: Path) -> None:
    contract_path, _, contract = _write_synthetic_bundle(tmp_path)
    contract["input_features"] = list(reversed(MODEL_FEATURE_COLUMNS))
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(DeploymentContractError, match="contract is invalid"):
        load_deployment_contract(contract_path)


def test_torch_base_version_removes_only_local_build_suffix() -> None:
    assert torch_base_version("2.14.0+cpu") == "2.14.0"
    assert torch_base_version("2.14.0") == "2.14.0"


@pytest.mark.integration
def test_official_state_group_train_bounds_match_deployment_contract() -> None:
    data_path = DEFAULT_RAW_DATA_DIR / "data.txt"
    if not data_path.is_file():
        pytest.skip("official UCI release is not available")
    frame = load_raw_dataset(data_path)
    split = build_dataset_splits(frame, seed=42)["state_group"]
    train_features = select_model_features(frame.iloc[split.train])
    contract = load_deployment_contract(DEFAULT_DEPLOYMENT_CONTRACT_PATH)

    assert len(split.train) == 8_352
    assert tuple(train_features.columns) == contract.input_features
    assert tuple(contract.continuous_feature_bounds) == SPEED_CENTERED_COLUMNS
    for feature in SPEED_CENTERED_COLUMNS:
        bounds = contract.continuous_feature_bounds[feature]
        assert float(train_features[feature].min()) == pytest.approx(
            bounds.minimum, rel=0.0, abs=1e-9
        )
        assert float(train_features[feature].max()) == pytest.approx(
            bounds.maximum, rel=0.0, abs=1e-9
        )
