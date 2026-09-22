"""Validation-only M3 selection and frozen-checkpoint final evaluation."""

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, cast

import numpy as np
import pandas as pd

from maritime_cbm.data.features import select_model_features
from maritime_cbm.data.schema import TARGET_COLUMNS
from maritime_cbm.data.splitting import (
    DatasetSplit,
    SplitMapping,
    build_dataset_splits,
    compute_split_hashes,
)
from maritime_cbm.modeling.metrics import (
    TargetRegressionMetrics,
    aggregated_state_group_selection_key,
    compute_regression_metrics,
)
from maritime_cbm.modeling.torch_models import (
    TorchCandidateSpec,
    build_torch_candidate_specs,
)
from maritime_cbm.modeling.torch_training import (
    DevicePreference,
    FittedTorchRegressor,
    LoadedTorchCheckpoint,
    TorchTrainingResult,
    TrainingConfig,
    fit_torch_regressor,
)

SELECTION_SCENARIOS: tuple[str, ...] = (
    "state_group",
    "compressor_holdout",
    "turbine_holdout",
)
FINAL_EVALUATION_SCENARIOS: tuple[str, ...] = (
    "random_row",
    "state_group",
    "compressor_holdout",
    "turbine_holdout",
)
SELECTION_SEED_COUNT = 3
type EvaluationRole = Literal["validation", "test"]
type TrainingSource = Literal["selection_checkpoint", "final_fit"]
type TorchSelectionKey = tuple[float, float, int, int, str]


def build_selection_seeds(base_seed: int) -> tuple[int, ...]:
    """Derive the frozen three-seed selection sequence from the project seed."""
    if base_seed < 0:
        raise ValueError("Base seed must be non-negative")
    return tuple(base_seed + offset for offset in range(SELECTION_SEED_COUNT))


@dataclass(slots=True)
class TorchValidationEvaluation:
    """One candidate, scenario and seed evaluated on validation only."""

    candidate: TorchCandidateSpec
    scenario: str
    seed: int
    validation_row_indices: np.ndarray
    training_result: TorchTrainingResult

    @property
    def metrics(self) -> tuple[TargetRegressionMetrics, ...]:
        """Return metrics captured at the restored best epoch."""
        return self.training_result.best_validation_metrics


@dataclass(frozen=True, slots=True)
class TorchSelectionOutcome:
    """Repeated-seed validation results selected without test-role access."""

    selected_candidate: TorchCandidateSpec
    selection_key: TorchSelectionKey
    evaluations: tuple[TorchValidationEvaluation, ...]
    selection_seeds: tuple[int, ...]
    split_hashes: dict[str, dict[str, str]]

    def selected_evaluation(self, scenario: str, seed: int) -> TorchValidationEvaluation:
        """Return one selected-candidate validation result for checkpoint storage."""
        matches = tuple(
            result
            for result in self.evaluations
            if result.candidate == self.selected_candidate
            and result.scenario == scenario
            and result.seed == seed
        )
        if len(matches) != 1:
            raise KeyError(f"No unique selected evaluation for scenario={scenario!r}, seed={seed}")
        return matches[0]


@dataclass(frozen=True, slots=True)
class TorchRoleEvaluation:
    """Predictions and metrics for one final-evaluation role."""

    role: EvaluationRole
    row_indices: np.ndarray
    predictions: np.ndarray
    metrics: tuple[TargetRegressionMetrics, ...]
    prediction_seconds: float


@dataclass(slots=True)
class TorchScenarioEvaluation:
    """One fixed M3 candidate evaluated within a split scenario."""

    candidate: TorchCandidateSpec
    scenario: str
    seed: int
    regressor: FittedTorchRegressor
    training_source: TrainingSource
    training_seconds: float
    best_epoch: int
    trainable_parameter_count: int
    roles: tuple[TorchRoleEvaluation, ...]

    def role(self, name: EvaluationRole) -> TorchRoleEvaluation:
        """Return one evaluated role by name."""
        matches = tuple(result for result in self.roles if result.role == name)
        if len(matches) != 1:
            raise KeyError(f"No unique role {name!r} for scenario {self.scenario!r}")
        return matches[0]


@dataclass(frozen=True, slots=True)
class TorchFinalEvaluationOutcome:
    """Frozen neural candidate evaluated once on every approved test scenario."""

    selected_candidate: TorchCandidateSpec
    evaluations: tuple[TorchScenarioEvaluation, ...]
    split_hashes: dict[str, dict[str, str]]

    @property
    def primary_regressor(self) -> FittedTorchRegressor:
        """Return the state-group checkpoint intended for later comparison work."""
        return next(
            result.regressor for result in self.evaluations if result.scenario == "state_group"
        )


def _fit_validation_only(
    features: pd.DataFrame,
    targets: np.ndarray,
    split: DatasetSplit,
    candidate: TorchCandidateSpec,
    *,
    seed: int,
    config: TrainingConfig,
    device: DevicePreference,
) -> TorchValidationEvaluation:
    result = fit_torch_regressor(
        features.iloc[split.train],
        targets[split.train],
        features.iloc[split.validation],
        targets[split.validation],
        candidate=candidate,
        seed=seed,
        config=config,
        device=device,
    )
    return TorchValidationEvaluation(
        candidate=candidate,
        scenario=split.name,
        seed=seed,
        validation_row_indices=split.validation.copy(),
        training_result=result,
    )


def _choose_torch_candidate(
    candidates: tuple[TorchCandidateSpec, ...],
    evaluations: tuple[TorchValidationEvaluation, ...],
) -> tuple[TorchCandidateSpec, TorchSelectionKey]:
    ranked = []
    for candidate in candidates:
        state_group_runs = tuple(
            result
            for result in evaluations
            if result.candidate == candidate and result.scenario == "state_group"
        )
        if not state_group_runs:
            raise ValueError(f"Candidate {candidate.candidate_id} has no state-group evaluations")
        parameter_counts = {
            result.training_result.trainable_parameter_count for result in state_group_runs
        }
        if len(parameter_counts) != 1:
            raise ValueError(f"Candidate {candidate.candidate_id} parameter count is inconsistent")
        key = aggregated_state_group_selection_key(
            tuple(result.metrics for result in state_group_runs),
            trainable_parameter_count=parameter_counts.pop(),
            architecture_rank=candidate.architecture_rank,
            candidate_id=candidate.candidate_id,
        )
        ranked.append((key, candidate))
    if not ranked:
        raise ValueError("At least one PyTorch candidate is required")
    selection_key, selected_candidate = min(ranked, key=lambda item: item[0])
    return selected_candidate, selection_key


def run_torch_model_selection(
    frame: pd.DataFrame,
    *,
    base_seed: int,
    candidates: tuple[TorchCandidateSpec, ...] | None = None,
    selection_seeds: tuple[int, ...] | None = None,
    config: TrainingConfig | None = None,
    device: DevicePreference = "cpu",
) -> TorchSelectionOutcome:
    """Fit the M3 grid on train roles and rank only state-group validation."""
    resolved_candidates = build_torch_candidate_specs() if candidates is None else candidates
    resolved_seeds = (
        build_selection_seeds(base_seed) if selection_seeds is None else selection_seeds
    )
    if not resolved_seeds:
        raise ValueError("At least one selection seed is required")
    resolved_config = TrainingConfig() if config is None else config
    features = select_model_features(frame)
    targets = frame.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float64, copy=True)
    splits = build_dataset_splits(frame, seed=base_seed)
    evaluations = tuple(
        _fit_validation_only(
            features,
            targets,
            splits[scenario],
            candidate,
            seed=seed,
            config=resolved_config,
            device=device,
        )
        for candidate in resolved_candidates
        for scenario in SELECTION_SCENARIOS
        for seed in resolved_seeds
    )
    selected_candidate, selection_key = _choose_torch_candidate(
        resolved_candidates,
        evaluations,
    )
    return TorchSelectionOutcome(
        selected_candidate=selected_candidate,
        selection_key=selection_key,
        evaluations=evaluations,
        selection_seeds=tuple(resolved_seeds),
        split_hashes=compute_split_hashes(splits),
    )


def _evaluate_roles(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    split: DatasetSplit,
    regressor: FittedTorchRegressor,
) -> tuple[TorchRoleEvaluation, ...]:
    targets = frame.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float64, copy=False)
    role_results = []
    for role in ("validation", "test"):
        indices = split.indices_by_role()[role]
        started = perf_counter()
        predictions = regressor.predict(features.iloc[indices])
        prediction_seconds = perf_counter() - started
        role_results.append(
            TorchRoleEvaluation(
                role=cast(EvaluationRole, role),
                row_indices=indices.copy(),
                predictions=predictions,
                metrics=compute_regression_metrics(targets[indices], predictions),
                prediction_seconds=prediction_seconds,
            )
        )
    return tuple(role_results)


def run_torch_final_evaluation(
    frame: pd.DataFrame,
    selected_candidate: TorchCandidateSpec,
    selection_checkpoints: Mapping[str, LoadedTorchCheckpoint],
    *,
    seed: int,
    config: TrainingConfig | None = None,
    device: DevicePreference = "cpu",
) -> TorchFinalEvaluationOutcome:
    """Evaluate fixed seed-42 checkpoints once and fit only the random-row scenario."""
    if set(selection_checkpoints) != set(SELECTION_SCENARIOS):
        raise ValueError(f"Selection checkpoints must cover exactly {SELECTION_SCENARIOS}")
    resolved_config = TrainingConfig() if config is None else config
    features = select_model_features(frame)
    targets = frame.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float64, copy=True)
    splits: SplitMapping = build_dataset_splits(frame, seed=seed)
    evaluations = []

    for scenario in FINAL_EVALUATION_SCENARIOS:
        split = splits[scenario]
        if scenario == "random_row":
            training_result = fit_torch_regressor(
                features.iloc[split.train],
                targets[split.train],
                features.iloc[split.validation],
                targets[split.validation],
                candidate=selected_candidate,
                seed=seed,
                config=resolved_config,
                device=device,
            )
            regressor = training_result.regressor
            training_source: TrainingSource = "final_fit"
            training_seconds = training_result.training_seconds
            best_epoch = training_result.best_epoch
            parameter_count = training_result.trainable_parameter_count
        else:
            checkpoint = selection_checkpoints[scenario]
            if checkpoint.regressor.candidate != selected_candidate:
                raise ValueError(f"Checkpoint candidate differs for scenario {scenario}")
            if checkpoint.seed != seed:
                raise ValueError(f"Checkpoint seed differs for scenario {scenario}")
            if checkpoint.training_config != resolved_config:
                raise ValueError(f"Checkpoint training config differs for scenario {scenario}")
            regressor = checkpoint.regressor
            training_source = "selection_checkpoint"
            training_seconds = checkpoint.training_seconds
            best_epoch = checkpoint.best_epoch
            parameter_count = checkpoint.trainable_parameter_count

        evaluations.append(
            TorchScenarioEvaluation(
                candidate=selected_candidate,
                scenario=scenario,
                seed=seed,
                regressor=regressor,
                training_source=training_source,
                training_seconds=training_seconds,
                best_epoch=best_epoch,
                trainable_parameter_count=parameter_count,
                roles=_evaluate_roles(frame, features, split, regressor),
            )
        )

    return TorchFinalEvaluationOutcome(
        selected_candidate=selected_candidate,
        evaluations=tuple(evaluations),
        split_hashes=compute_split_hashes(splits),
    )
