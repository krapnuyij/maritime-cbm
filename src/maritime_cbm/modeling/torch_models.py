"""Frozen M3 PyTorch candidate grid and small multi-output regressors."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from maritime_cbm.data.schema import MODEL_FEATURE_COLUMNS, TARGET_COLUMNS

HIDDEN_LAYER_OPTIONS: tuple[tuple[int, int], ...] = ((64, 32), (128, 64))
type TorchArchitecture = Literal["mlp", "linear_residual_mlp"]
type TorchPreprocessing = Literal["raw", "speed_centered"]


@dataclass(frozen=True, slots=True)
class TorchCandidateSpec:
    """Serializable definition of one pre-approved M3 neural candidate."""

    candidate_id: str
    architecture: TorchArchitecture
    preprocessing: TorchPreprocessing
    hidden_sizes: tuple[int, int]
    architecture_rank: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON- and checkpoint-safe representation."""
        return {
            "candidate_id": self.candidate_id,
            "architecture": self.architecture,
            "preprocessing": self.preprocessing,
            "hidden_sizes": list(self.hidden_sizes),
            "architecture_rank": self.architecture_rank,
        }


def _hidden_token(hidden_sizes: tuple[int, int]) -> str:
    return "_".join(str(size) for size in hidden_sizes)


def build_torch_candidate_specs() -> tuple[TorchCandidateSpec, ...]:
    """Return the frozen six-candidate M3 grid in deterministic order."""
    candidates = []
    families: tuple[tuple[TorchArchitecture, TorchPreprocessing, int, str], ...] = (
        ("mlp", "raw", 1, "mlp_raw"),
        ("mlp", "speed_centered", 2, "mlp_speed_centered"),
        (
            "linear_residual_mlp",
            "speed_centered",
            3,
            "linear_residual_mlp_speed_centered",
        ),
    )
    for architecture, preprocessing, rank, prefix in families:
        for hidden_sizes in HIDDEN_LAYER_OPTIONS:
            candidates.append(
                TorchCandidateSpec(
                    candidate_id=f"{prefix}_hidden_{_hidden_token(hidden_sizes)}",
                    architecture=architecture,
                    preprocessing=preprocessing,
                    hidden_sizes=hidden_sizes,
                    architecture_rank=rank,
                )
            )
    return tuple(candidates)


def get_torch_candidate_spec(candidate_id: str) -> TorchCandidateSpec:
    """Return one frozen candidate by stable identifier."""
    matches = tuple(
        spec for spec in build_torch_candidate_specs() if spec.candidate_id == candidate_id
    )
    if not matches:
        raise KeyError(f"Unknown PyTorch candidate: {candidate_id}")
    return matches[0]


def torch_candidate_from_dict(payload: object) -> TorchCandidateSpec:
    """Resolve and verify a serialized candidate against the frozen grid."""
    if not isinstance(payload, dict):
        raise ValueError("PyTorch candidate payload must be a dictionary")
    candidate_id = payload.get("candidate_id")
    if not isinstance(candidate_id, str):
        raise ValueError("PyTorch candidate payload has no valid candidate_id")
    try:
        candidate = get_torch_candidate_spec(candidate_id)
    except KeyError as error:
        raise ValueError(str(error)) from error
    if payload != candidate.to_dict():
        raise ValueError("PyTorch candidate payload differs from the approved grid")
    return candidate


def _mlp_layers(input_size: int, hidden_sizes: tuple[int, int], output_size: int) -> nn.Sequential:
    first_hidden, second_hidden = hidden_sizes
    return nn.Sequential(
        nn.Linear(input_size, first_hidden),
        nn.ReLU(),
        nn.Linear(first_hidden, second_hidden),
        nn.ReLU(),
        nn.Linear(second_hidden, output_size),
    )


class MLPRegressor(nn.Module):
    """Two-hidden-layer MLP with an unclipped linear output."""

    def __init__(self, input_size: int, hidden_sizes: tuple[int, int], output_size: int) -> None:
        super().__init__()
        self.network = _mlp_layers(input_size, hidden_sizes, output_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return standardized degradation-coefficient predictions."""
        return self.network(features)


class LinearResidualMLPRegressor(nn.Module):
    """Affine extrapolation path plus a zero-initialized nonlinear correction."""

    def __init__(self, input_size: int, hidden_sizes: tuple[int, int], output_size: int) -> None:
        super().__init__()
        self.linear_path = nn.Linear(input_size, output_size)
        self.residual_path = _mlp_layers(input_size, hidden_sizes, output_size)
        residual_output = self.residual_path[-1]
        if not isinstance(residual_output, nn.Linear):  # pragma: no cover - closed construction.
            raise TypeError("Residual output layer must be linear")
        nn.init.zeros_(residual_output.weight)
        nn.init.zeros_(residual_output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Add the trainable affine path and nonlinear residual."""
        return self.linear_path(features) + self.residual_path(features)


def build_torch_model(spec: TorchCandidateSpec) -> nn.Module:
    """Construct an unfitted M3 model for one candidate specification."""
    input_size = len(MODEL_FEATURE_COLUMNS)
    output_size = len(TARGET_COLUMNS)
    if spec.architecture == "mlp":
        return MLPRegressor(input_size, spec.hidden_sizes, output_size)
    if spec.architecture == "linear_residual_mlp":
        return LinearResidualMLPRegressor(input_size, spec.hidden_sizes, output_size)
    raise ValueError(f"Unsupported PyTorch architecture: {spec.architecture}")


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of parameters updated by the optimizer."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
