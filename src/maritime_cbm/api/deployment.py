"""Validated deployment contract and M3 inference bundle loading."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from maritime_cbm.alerting.policy import (
    ALERT_SEVERITY_THRESHOLD,
    WATCH_SEVERITY_THRESHOLD,
)
from maritime_cbm.data.loader import compute_sha256
from maritime_cbm.data.schema import MODEL_FEATURE_COLUMNS, TARGET_COLUMNS
from maritime_cbm.modeling.preprocessing import SPEED_CENTERED_COLUMNS
from maritime_cbm.modeling.torch_training import LoadedTorchCheckpoint, load_torch_checkpoint

DEPLOYMENT_SCHEMA_VERSION = 1
DEPLOYMENT_CANDIDATE_ID = "linear_residual_mlp_speed_centered_hidden_128_64"
DEPLOYMENT_MODEL_VERSION = "m3-linear-residual-mlp-v1"
DEPLOYMENT_POLICY_VERSION = "m4-severity-policy-v1"


class DeploymentContractError(RuntimeError):
    """Raised when a deployment contract and its checkpoint are incompatible."""


class FeatureBounds(BaseModel):
    """Inclusive observed bounds for one continuous sensor feature."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    minimum: float
    maximum: float

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        """Require an ordered, non-empty inclusive interval."""
        if self.minimum > self.maximum:
            raise ValueError("minimum must not exceed maximum")
        return self


class DeploymentContract(BaseModel):
    """Version-controlled model, input and alert-policy deployment contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal[1]
    model_version: str
    candidate_id: str
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_seed: int = Field(ge=0)
    checkpoint_scenario: Literal["state_group"]
    torch_base_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    input_features: tuple[str, ...]
    target_columns: tuple[str, ...]
    allowed_speeds: tuple[float, ...]
    continuous_feature_bounds: dict[str, FeatureBounds]
    policy_version: str
    watch_severity_threshold: float = Field(ge=0.0, le=1.0)
    alert_severity_threshold: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_supported_contract(self) -> Self:
        """Reject drift from the model, schema and policy implemented by this release."""
        if self.model_version != DEPLOYMENT_MODEL_VERSION:
            raise ValueError("unsupported deployment model version")
        if self.candidate_id != DEPLOYMENT_CANDIDATE_ID:
            raise ValueError("unsupported deployment candidate")
        if self.policy_version != DEPLOYMENT_POLICY_VERSION:
            raise ValueError("unsupported alert policy version")
        if self.input_features != MODEL_FEATURE_COLUMNS:
            raise ValueError("input feature order differs from the model schema")
        if self.target_columns != TARGET_COLUMNS:
            raise ValueError("target order differs from the model schema")
        if tuple(self.continuous_feature_bounds) != SPEED_CENTERED_COLUMNS:
            raise ValueError("continuous feature bounds differ from the model schema")
        if len(self.allowed_speeds) == 0 or len(set(self.allowed_speeds)) != len(
            self.allowed_speeds
        ):
            raise ValueError("allowed speeds must be non-empty and unique")
        if tuple(sorted(self.allowed_speeds)) != self.allowed_speeds:
            raise ValueError("allowed speeds must be sorted")
        if self.watch_severity_threshold >= self.alert_severity_threshold:
            raise ValueError("watch threshold must be lower than alert threshold")
        if not np.isclose(
            self.watch_severity_threshold,
            WATCH_SEVERITY_THRESHOLD,
            rtol=0.0,
            atol=0.0,
        ) or not np.isclose(
            self.alert_severity_threshold,
            ALERT_SEVERITY_THRESHOLD,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("alert thresholds differ from the implemented M4 policy")
        return self


@dataclass(frozen=True, slots=True)
class LoadedDeployment:
    """Validated immutable deployment contract and loaded M3 checkpoint."""

    contract: DeploymentContract
    checkpoint: LoadedTorchCheckpoint
    checkpoint_path: Path


def torch_base_version(version: object) -> str:
    """Return the public PyTorch version without a local build suffix."""
    value = str(version)
    base = value.split("+", maxsplit=1)[0]
    if not base:
        raise DeploymentContractError("PyTorch version is empty")
    return base


def load_deployment_contract(path: Path) -> DeploymentContract:
    """Load and validate the version-controlled JSON deployment contract."""
    if not path.is_file():
        raise DeploymentContractError("Deployment contract file is missing")
    try:
        return DeploymentContract.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DeploymentContractError("Deployment contract is invalid") from error


def _validate_checkpoint_contract(
    contract: DeploymentContract,
    checkpoint: LoadedTorchCheckpoint,
) -> None:
    candidate = checkpoint.regressor.candidate
    if candidate.candidate_id != contract.candidate_id:
        raise DeploymentContractError("Checkpoint candidate differs from the contract")
    if checkpoint.seed != contract.checkpoint_seed:
        raise DeploymentContractError("Checkpoint seed differs from the contract")
    metadata = checkpoint.metadata
    if (
        metadata.get("scenario") != contract.checkpoint_scenario
        or metadata.get("role") != "selection_validation"
        or metadata.get("seed") != contract.checkpoint_seed
    ):
        raise DeploymentContractError("Checkpoint state-group metadata differs from the contract")

    speed_values = checkpoint.regressor.preprocessor.speed_values
    if speed_values.shape != (len(contract.allowed_speeds),) or not np.allclose(
        speed_values,
        np.asarray(contract.allowed_speeds, dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    ):
        raise DeploymentContractError("Checkpoint speed values differ from the contract")

    checkpoint_version = torch_base_version(checkpoint.torch_version)
    runtime_version = torch_base_version(torch.__version__)
    if len({contract.torch_base_version, checkpoint_version, runtime_version}) != 1:
        raise DeploymentContractError(
            "Contract, checkpoint and runtime PyTorch base versions differ"
        )


def load_deployment(contract_path: Path, checkpoint_path: Path) -> LoadedDeployment:
    """Verify artifact integrity and return the CPU-ready deployment bundle."""
    contract = load_deployment_contract(contract_path)
    if not checkpoint_path.is_file():
        raise DeploymentContractError("Deployment checkpoint file is missing")
    if compute_sha256(checkpoint_path) != contract.checkpoint_sha256:
        raise DeploymentContractError("Deployment checkpoint SHA-256 differs from the contract")
    try:
        checkpoint = load_torch_checkpoint(checkpoint_path)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise DeploymentContractError("Deployment checkpoint could not be loaded") from error
    _validate_checkpoint_contract(contract, checkpoint)
    return LoadedDeployment(
        contract=contract,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
    )
