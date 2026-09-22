"""Regression metrics and the approved state-group model selection score."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from maritime_cbm.data.schema import EXPECTED_KMC_VALUES, EXPECTED_KMT_VALUES, TARGET_COLUMNS

TARGET_BOUNDS: dict[str, tuple[float, float]] = {
    "kMc": (min(EXPECTED_KMC_VALUES), max(EXPECTED_KMC_VALUES)),
    "kMt": (min(EXPECTED_KMT_VALUES), max(EXPECTED_KMT_VALUES)),
}
TARGET_RANGES: dict[str, float] = {
    target: upper - lower for target, (lower, upper) in TARGET_BOUNDS.items()
}


class MetricInputError(ValueError):
    """Raised when regression targets or predictions violate the M2 contract."""


@dataclass(frozen=True, slots=True)
class TargetRegressionMetrics:
    """Metrics for one degradation coefficient in its original units."""

    target: str
    sample_count: int
    mae: float
    rmse: float
    r2: float
    bias: float
    absolute_error_p95: float
    below_range_count: int
    above_range_count: int
    nrmse: float

    def to_dict(self) -> dict[str, str | int | float]:
        """Return a serialization-ready metric row."""
        return asdict(self)


def _target_array(values: object, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    expected_shape_suffix = (len(TARGET_COLUMNS),)
    if array.ndim != 2 or array.shape[1:] != expected_shape_suffix or len(array) == 0:
        raise MetricInputError(
            f"{name} must have shape (n_samples, {len(TARGET_COLUMNS)}) with at least one row"
        )
    if not np.isfinite(array).all():
        raise MetricInputError(f"{name} must contain only finite values")
    return array


def compute_regression_metrics(
    y_true: object,
    y_pred: object,
) -> tuple[TargetRegressionMetrics, ...]:
    """Compute per-target errors without clipping predictions."""
    true_values = _target_array(y_true, "y_true")
    predicted_values = _target_array(y_pred, "y_pred")
    if true_values.shape != predicted_values.shape:
        raise MetricInputError(
            f"y_true and y_pred shapes differ: {true_values.shape} != {predicted_values.shape}"
        )

    mae_values = np.asarray(
        mean_absolute_error(true_values, predicted_values, multioutput="raw_values")
    )
    rmse_values = np.asarray(
        root_mean_squared_error(true_values, predicted_values, multioutput="raw_values")
    )
    r2_values = np.asarray(r2_score(true_values, predicted_values, multioutput="raw_values"))
    errors = predicted_values - true_values

    rows = []
    for index, target in enumerate(TARGET_COLUMNS):
        lower, upper = TARGET_BOUNDS[target]
        target_errors = errors[:, index]
        rmse = float(rmse_values[index])
        rows.append(
            TargetRegressionMetrics(
                target=target,
                sample_count=len(true_values),
                mae=float(mae_values[index]),
                rmse=rmse,
                r2=float(r2_values[index]),
                bias=float(np.mean(target_errors)),
                absolute_error_p95=float(np.percentile(np.abs(target_errors), 95)),
                below_range_count=int(np.count_nonzero(predicted_values[:, index] < lower)),
                above_range_count=int(np.count_nonzero(predicted_values[:, index] > upper)),
                nrmse=rmse / TARGET_RANGES[target],
            )
        )
    return tuple(rows)


def state_group_selection_key(
    metrics: tuple[TargetRegressionMetrics, ...],
    *,
    complexity_rank: int,
    candidate_id: str,
) -> tuple[float, float, int, str]:
    """Return the raw-float deterministic key approved for M2 model selection."""
    by_target = {metric.target: metric for metric in metrics}
    if set(by_target) != set(TARGET_COLUMNS):
        raise MetricInputError(
            f"Expected metrics for targets {TARGET_COLUMNS}, found {set(by_target)}"
        )
    nrmse_values = tuple(by_target[target].nrmse for target in TARGET_COLUMNS)
    return (
        float(np.mean(nrmse_values)),
        float(max(nrmse_values)),
        complexity_rank,
        candidate_id,
    )


def aggregated_state_group_selection_key(
    metric_runs: Sequence[tuple[TargetRegressionMetrics, ...]],
    *,
    trainable_parameter_count: int,
    architecture_rank: int,
    candidate_id: str,
) -> tuple[float, float, int, int, str]:
    """Rank a stochastic model after averaging each target across repeated seeds."""
    if not metric_runs:
        raise MetricInputError("At least one metric run is required for aggregate selection")
    if trainable_parameter_count < 0:
        raise MetricInputError("Trainable parameter count must be non-negative")
    if architecture_rank < 0:
        raise MetricInputError("Architecture rank must be non-negative")

    nrmse_by_target = {target: [] for target in TARGET_COLUMNS}
    for metrics in metric_runs:
        by_target = {metric.target: metric for metric in metrics}
        if len(by_target) != len(metrics) or set(by_target) != set(TARGET_COLUMNS):
            raise MetricInputError(
                f"Expected one metric per target {TARGET_COLUMNS}, found "
                f"{tuple(metric.target for metric in metrics)}"
            )
        for target in TARGET_COLUMNS:
            nrmse_by_target[target].append(by_target[target].nrmse)

    target_means = tuple(
        float(np.mean(nrmse_by_target[target], dtype=np.float64)) for target in TARGET_COLUMNS
    )
    return (
        float(np.mean(target_means, dtype=np.float64)),
        float(max(target_means)),
        trainable_parameter_count,
        architecture_rank,
        candidate_id,
    )
