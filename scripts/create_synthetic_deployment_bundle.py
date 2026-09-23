"""Create an ephemeral synthetic deployment bundle for Docker smoke tests only."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from maritime_cbm.api.deployment import (
    DEPLOYMENT_CANDIDATE_ID,
    DEPLOYMENT_MODEL_VERSION,
    DEPLOYMENT_POLICY_VERSION,
    DeploymentContract,
    FeatureBounds,
    torch_base_version,
)
from maritime_cbm.data.loader import compute_sha256
from maritime_cbm.data.schema import MODEL_FEATURE_COLUMNS, TARGET_COLUMNS
from maritime_cbm.modeling.preprocessing import SPEED_CENTERED_COLUMNS
from maritime_cbm.modeling.torch_models import get_torch_candidate_spec
from maritime_cbm.modeling.torch_training import (
    TrainingConfig,
    fit_torch_regressor,
    save_torch_checkpoint,
)


def _training_data() -> tuple[pd.DataFrame, np.ndarray]:
    row_count = 27
    values = np.arange(row_count * len(MODEL_FEATURE_COLUMNS), dtype=np.float64).reshape(
        row_count, len(MODEL_FEATURE_COLUMNS)
    )
    frame = pd.DataFrame(values / 100.0 + 1.0, columns=MODEL_FEATURE_COLUMNS)
    frame.loc[:, "v"] = np.tile(np.asarray([3.0, 6.0, 9.0]), 9)
    targets = np.column_stack(
        (
            0.975 + frame["T2"].to_numpy() * 1e-4,
            0.9875 + frame["P2"].to_numpy() * 5e-5,
        )
    )
    return frame, targets


def create_bundle(output_directory: Path) -> tuple[Path, Path]:
    """Write a deterministic-shape, test-only contract and checkpoint bundle."""
    output_directory.mkdir(parents=True, exist_ok=True)
    features, targets = _training_data()
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
    checkpoint_path = output_directory / "synthetic-smoke.pt"
    save_torch_checkpoint(
        result,
        checkpoint_path,
        metadata={
            "scenario": "state_group",
            "role": "selection_validation",
            "seed": 42,
            "synthetic_test_only": True,
        },
    )
    contract = DeploymentContract(
        schema_version=1,
        model_version=DEPLOYMENT_MODEL_VERSION,
        candidate_id=DEPLOYMENT_CANDIDATE_ID,
        checkpoint_sha256=compute_sha256(checkpoint_path),
        checkpoint_seed=42,
        checkpoint_scenario="state_group",
        torch_base_version=torch_base_version(torch.__version__),
        input_features=MODEL_FEATURE_COLUMNS,
        target_columns=TARGET_COLUMNS,
        allowed_speeds=(3.0, 6.0, 9.0),
        continuous_feature_bounds={
            feature: FeatureBounds(minimum=0.0, maximum=100.0) for feature in SPEED_CENTERED_COLUMNS
        },
        policy_version=DEPLOYMENT_POLICY_VERSION,
        watch_severity_threshold=0.5,
        alert_severity_threshold=0.8,
    )
    contract_path = output_directory / "synthetic-contract.json"
    contract_path.write_text(contract.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return contract_path, checkpoint_path


def main() -> None:
    """Parse the output directory and create a smoke-test-only bundle."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    contract_path, checkpoint_path = create_bundle(arguments.output_directory)
    print(f"synthetic_contract={contract_path}")
    print(f"synthetic_checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
