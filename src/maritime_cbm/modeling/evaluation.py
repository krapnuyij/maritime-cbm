"""Validation-only selection and frozen-candidate final evaluation."""

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor

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
    compute_regression_metrics,
    state_group_selection_key,
)
from maritime_cbm.modeling.models import CandidateSpec, build_candidate_specs, build_estimator

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
type EvaluationRole = Literal["validation", "test"]


@dataclass(frozen=True, slots=True)
class RoleEvaluation:
    """Predictions and metrics for one scenario role."""

    role: EvaluationRole
    row_indices: np.ndarray
    predictions: np.ndarray
    metrics: tuple[TargetRegressionMetrics, ...]
    prediction_seconds: float


@dataclass(frozen=True, slots=True)
class ScenarioEvaluation:
    """One fitted candidate evaluated within a single split scenario."""

    candidate: CandidateSpec
    scenario: str
    estimator: TransformedTargetRegressor
    fit_seconds: float
    roles: tuple[RoleEvaluation, ...]

    def role(self, name: EvaluationRole) -> RoleEvaluation:
        """Return one evaluated role by name."""
        matches = tuple(result for result in self.roles if result.role == name)
        if not matches:
            raise KeyError(f"Role {name!r} was not evaluated for scenario {self.scenario!r}")
        return matches[0]


@dataclass(frozen=True, slots=True)
class SelectionOutcome:
    """Validation results and the candidate selected without test access."""

    selected_candidate: CandidateSpec
    selection_key: tuple[float, float, int, str]
    evaluations: tuple[ScenarioEvaluation, ...]
    split_hashes: dict[str, dict[str, str]]


@dataclass(frozen=True, slots=True)
class FinalEvaluationOutcome:
    """Validation and one-time test results for the frozen selected candidate."""

    selected_candidate: CandidateSpec
    evaluations: tuple[ScenarioEvaluation, ...]
    split_hashes: dict[str, dict[str, str]]

    @property
    def primary_estimator(self) -> TransformedTargetRegressor:
        """Return the fitted state-group estimator intended for later service work."""
        return next(
            result.estimator for result in self.evaluations if result.scenario == "state_group"
        )


def _fit_and_evaluate(
    frame: pd.DataFrame,
    split: DatasetSplit,
    candidate: CandidateSpec,
    *,
    roles: tuple[EvaluationRole, ...],
    seed: int,
) -> ScenarioEvaluation:
    features = select_model_features(frame)
    targets = frame.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float64, copy=True)
    estimator = build_estimator(candidate, seed=seed)

    fit_started = perf_counter()
    estimator.fit(features.iloc[split.train], targets[split.train])
    fit_seconds = perf_counter() - fit_started

    role_results = []
    indices_by_role = split.indices_by_role()
    for role in roles:
        indices = indices_by_role[role]
        prediction_started = perf_counter()
        predictions = np.asarray(estimator.predict(features.iloc[indices]), dtype=np.float64)
        prediction_seconds = perf_counter() - prediction_started
        role_results.append(
            RoleEvaluation(
                role=role,
                row_indices=indices.copy(),
                predictions=predictions,
                metrics=compute_regression_metrics(targets[indices], predictions),
                prediction_seconds=prediction_seconds,
            )
        )

    return ScenarioEvaluation(
        candidate=candidate,
        scenario=split.name,
        estimator=estimator,
        fit_seconds=fit_seconds,
        roles=tuple(role_results),
    )


def _choose_candidate(
    candidates: tuple[CandidateSpec, ...],
    evaluations: tuple[ScenarioEvaluation, ...],
) -> tuple[CandidateSpec, tuple[float, float, int, str]]:
    keys = []
    for candidate in candidates:
        if not candidate.selectable:
            continue
        state_group_result = next(
            result
            for result in evaluations
            if result.candidate == candidate and result.scenario == "state_group"
        )
        key = state_group_selection_key(
            state_group_result.role("validation").metrics,
            complexity_rank=candidate.complexity_rank,
            candidate_id=candidate.candidate_id,
        )
        keys.append((key, candidate))
    if not keys:
        raise ValueError("At least one selectable candidate is required")
    selection_key, selected_candidate = min(keys, key=lambda item: item[0])
    return selected_candidate, selection_key


def run_model_selection(
    frame: pd.DataFrame,
    *,
    seed: int,
    candidates: tuple[CandidateSpec, ...] | None = None,
) -> SelectionOutcome:
    """Fit the candidate grid on train and select using validation results only."""
    resolved_candidates = build_candidate_specs() if candidates is None else candidates
    splits = build_dataset_splits(frame, seed=seed)
    evaluations = tuple(
        _fit_and_evaluate(
            frame,
            splits[scenario],
            candidate,
            roles=("validation",),
            seed=seed,
        )
        for candidate in resolved_candidates
        for scenario in SELECTION_SCENARIOS
    )
    selected_candidate, selection_key = _choose_candidate(resolved_candidates, evaluations)
    return SelectionOutcome(
        selected_candidate=selected_candidate,
        selection_key=selection_key,
        evaluations=evaluations,
        split_hashes=compute_split_hashes(splits),
    )


def run_final_evaluation(
    frame: pd.DataFrame,
    selected_candidate: CandidateSpec,
    *,
    seed: int,
) -> FinalEvaluationOutcome:
    """Evaluate a previously frozen candidate once on every approved test scenario."""
    splits: SplitMapping = build_dataset_splits(frame, seed=seed)
    evaluations = tuple(
        _fit_and_evaluate(
            frame,
            splits[scenario],
            selected_candidate,
            roles=("validation", "test"),
            seed=seed,
        )
        for scenario in FINAL_EVALUATION_SCENARIOS
    )
    return FinalEvaluationOutcome(
        selected_candidate=selected_candidate,
        evaluations=evaluations,
        split_hashes=compute_split_hashes(splits),
    )
