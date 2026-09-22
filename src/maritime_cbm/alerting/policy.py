"""Normalized degradation severity and fixed PoC alert states."""

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from maritime_cbm.data.schema import TARGET_COLUMNS
from maritime_cbm.modeling.metrics import TARGET_RANGES

WATCH_SEVERITY_THRESHOLD = 0.5
ALERT_SEVERITY_THRESHOLD = 0.8
SENSITIVITY_THRESHOLDS: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)
ALERT_CHANNELS: tuple[str, ...] = (*TARGET_COLUMNS, "any")

type AlertChannel = Literal["kMc", "kMt", "any"]
type AlertState = Literal["normal", "watch", "alert"]


class AlertPolicyInputError(ValueError):
    """Raised when coefficient or severity values violate the policy contract."""


def _two_target_array(values: object, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(TARGET_COLUMNS) or len(array) == 0:
        raise AlertPolicyInputError(
            f"{name} must have shape (n_samples, {len(TARGET_COLUMNS)}) with at least one row"
        )
    if not np.isfinite(array).all():
        raise AlertPolicyInputError(f"{name} must contain only finite values")
    return array


def _one_dimensional_scores(values: object, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise AlertPolicyInputError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise AlertPolicyInputError(f"{name} must contain only finite values")
    return array


def validate_severity_threshold(value: float, *, name: str = "severity threshold") -> float:
    """Validate a normalized policy threshold without restricting prediction scores."""
    threshold = float(value)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise AlertPolicyInputError(f"{name} must be finite and within [0, 1]")
    return threshold


def validate_decision_threshold(value: float, *, name: str = "decision threshold") -> float:
    """Validate an unclipped decision cutoff derived from prediction scores."""
    threshold = float(value)
    if not np.isfinite(threshold):
        raise AlertPolicyInputError(f"{name} must be finite")
    return threshold


def coefficients_to_severity(values: object) -> np.ndarray:
    """Convert kMc and kMt coefficients to unclipped normalized degradation severity."""
    coefficients = _two_target_array(values, name="coefficients")
    denominators = np.asarray(
        [TARGET_RANGES[target] for target in TARGET_COLUMNS],
        dtype=np.float64,
    )
    return (1.0 - coefficients) / denominators


def channel_severity(values: object, channel: AlertChannel) -> np.ndarray:
    """Return one component severity or the worst component severity per row."""
    severity = _two_target_array(values, name="severity")
    if channel == "any":
        return np.max(severity, axis=1)
    if channel not in TARGET_COLUMNS:
        raise AlertPolicyInputError(f"Unsupported alert channel: {channel}")
    return severity[:, TARGET_COLUMNS.index(channel)]


def severity_labels(values: object, *, threshold: float = ALERT_SEVERITY_THRESHOLD) -> np.ndarray:
    """Return inclusive binary labels for one severity score vector."""
    scores = _one_dimensional_scores(values, name="severity scores")
    resolved_threshold = validate_decision_threshold(threshold)
    return scores >= resolved_threshold


def severity_states(
    values: object,
    *,
    watch_threshold: float = WATCH_SEVERITY_THRESHOLD,
    alert_threshold: float = ALERT_SEVERITY_THRESHOLD,
) -> np.ndarray:
    """Assign normal, watch or alert without clipping out-of-range severity."""
    scores = _one_dimensional_scores(values, name="severity scores")
    resolved_watch = validate_severity_threshold(watch_threshold, name="watch threshold")
    resolved_alert = validate_severity_threshold(alert_threshold, name="alert threshold")
    if resolved_watch >= resolved_alert:
        raise AlertPolicyInputError("Watch threshold must be lower than alert threshold")
    states = np.full(len(scores), "normal", dtype="<U6")
    states[scores >= resolved_watch] = "watch"
    states[scores >= resolved_alert] = "alert"
    return states


@dataclass(frozen=True, slots=True)
class AlertPolicyDecision:
    """Component and overall severity states for one prediction batch."""

    severity: np.ndarray
    compressor_state: np.ndarray
    turbine_state: np.ndarray
    overall_state: np.ndarray


def evaluate_alert_policy(
    coefficients: object,
    *,
    watch_threshold: float = WATCH_SEVERITY_THRESHOLD,
    alert_threshold: float = ALERT_SEVERITY_THRESHOLD,
) -> AlertPolicyDecision:
    """Evaluate component and worst-component states from regression coefficients."""
    severity = coefficients_to_severity(coefficients)
    compressor = severity_states(
        channel_severity(severity, "kMc"),
        watch_threshold=watch_threshold,
        alert_threshold=alert_threshold,
    )
    turbine = severity_states(
        channel_severity(severity, "kMt"),
        watch_threshold=watch_threshold,
        alert_threshold=alert_threshold,
    )
    overall = severity_states(
        channel_severity(severity, "any"),
        watch_threshold=watch_threshold,
        alert_threshold=alert_threshold,
    )
    return AlertPolicyDecision(
        severity=severity,
        compressor_state=compressor,
        turbine_state=turbine,
        overall_state=overall,
    )


def parse_alert_channel(value: str) -> AlertChannel:
    """Validate a serialized alert channel."""
    if value not in ALERT_CHANNELS:
        raise AlertPolicyInputError(f"Unsupported alert channel: {value}")
    return cast(AlertChannel, value)
