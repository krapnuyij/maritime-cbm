"""Single-class-safe alert metrics and validation-only FPR cutoff selection."""

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score

from maritime_cbm.alerting.policy import (
    ALERT_SEVERITY_THRESHOLD,
    AlertPolicyInputError,
    severity_labels,
    validate_decision_threshold,
    validate_severity_threshold,
)


def _score_pair(reference: object, predicted: object) -> tuple[np.ndarray, np.ndarray]:
    reference_scores = np.asarray(reference, dtype=np.float64)
    predicted_scores = np.asarray(predicted, dtype=np.float64)
    if reference_scores.ndim != 1 or predicted_scores.ndim != 1:
        raise AlertPolicyInputError("Reference and predicted severity must be one-dimensional")
    if not len(reference_scores) or reference_scores.shape != predicted_scores.shape:
        raise AlertPolicyInputError(
            "Reference and predicted severity must have the same non-empty shape"
        )
    if not np.isfinite(reference_scores).all() or not np.isfinite(predicted_scores).all():
        raise AlertPolicyInputError("Reference and predicted severity must be finite")
    return reference_scores, predicted_scores


@dataclass(frozen=True, slots=True)
class AlertClassificationMetrics:
    """Binary alert metrics with explicit single-class availability metadata."""

    sample_count: int
    positive_count: int
    negative_count: int
    predicted_positive_count: int
    tp: int
    fp: int
    tn: int
    fn: int
    reference_threshold: float
    decision_threshold: float
    precision: float | None
    recall: float | None
    f1: float | None
    fpr: float | None
    miss_rate: float | None
    pr_auc: float | None
    reference_prevalence: float
    metric_valid: bool
    unavailable_reason: str | None

    def to_dict(self) -> dict[str, str | bool | int | float | None]:
        """Return a stable serialization-ready metric row."""
        return asdict(self)


def compute_alert_metrics(
    reference_severity: object,
    predicted_severity: object,
    *,
    reference_threshold: float = ALERT_SEVERITY_THRESHOLD,
    decision_threshold: float | None = None,
) -> AlertClassificationMetrics:
    """Compute policy metrics without treating single-class PR-AUC as discrimination."""
    reference_scores, predicted_scores = _score_pair(
        reference_severity,
        predicted_severity,
    )
    resolved_reference = validate_severity_threshold(
        reference_threshold,
        name="reference threshold",
    )
    resolved_decision = (
        resolved_reference
        if decision_threshold is None
        else validate_decision_threshold(decision_threshold)
    )
    reference_labels = severity_labels(reference_scores, threshold=resolved_reference)
    predicted_labels = severity_labels(predicted_scores, threshold=resolved_decision)

    tp = int(np.count_nonzero(reference_labels & predicted_labels))
    fn = int(np.count_nonzero(reference_labels & ~predicted_labels))
    fp = int(np.count_nonzero(~reference_labels & predicted_labels))
    tn = int(np.count_nonzero(~reference_labels & ~predicted_labels))
    positive_count = tp + fn
    negative_count = fp + tn
    predicted_positive_count = tp + fp
    sample_count = len(reference_labels)
    prevalence = positive_count / sample_count

    if negative_count == 0:
        return AlertClassificationMetrics(
            sample_count=sample_count,
            positive_count=positive_count,
            negative_count=negative_count,
            predicted_positive_count=predicted_positive_count,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            reference_threshold=resolved_reference,
            decision_threshold=resolved_decision,
            precision=None,
            recall=tp / positive_count,
            f1=None,
            fpr=None,
            miss_rate=fn / positive_count,
            pr_auc=None,
            reference_prevalence=prevalence,
            metric_valid=False,
            unavailable_reason="single_class_all_positive",
        )
    if positive_count == 0:
        return AlertClassificationMetrics(
            sample_count=sample_count,
            positive_count=positive_count,
            negative_count=negative_count,
            predicted_positive_count=predicted_positive_count,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            reference_threshold=resolved_reference,
            decision_threshold=resolved_decision,
            precision=None,
            recall=None,
            f1=None,
            fpr=fp / negative_count,
            miss_rate=None,
            pr_auc=None,
            reference_prevalence=prevalence,
            metric_valid=False,
            unavailable_reason="single_class_all_negative",
        )

    precision = tp / predicted_positive_count if predicted_positive_count else 0.0
    recall = tp / positive_count
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return AlertClassificationMetrics(
        sample_count=sample_count,
        positive_count=positive_count,
        negative_count=negative_count,
        predicted_positive_count=predicted_positive_count,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        reference_threshold=resolved_reference,
        decision_threshold=resolved_decision,
        precision=precision,
        recall=recall,
        f1=f1,
        fpr=fp / negative_count,
        miss_rate=fn / positive_count,
        pr_auc=float(average_precision_score(reference_labels, predicted_scores)),
        reference_prevalence=prevalence,
        metric_valid=True,
        unavailable_reason=None,
    )


@dataclass(frozen=True, slots=True)
class FixedFprCutoff:
    """A deterministic validation cutoff selected under an empirical FPR limit."""

    target_fpr: float
    cutoff: float
    validation_recall: float
    validation_fpr: float
    validation_precision: float

    def to_dict(self) -> dict[str, float]:
        """Return a stable serialization-ready cutoff row."""
        return asdict(self)


def select_fixed_fpr_cutoff(
    reference_severity: object,
    predicted_severity: object,
    *,
    reference_threshold: float = ALERT_SEVERITY_THRESHOLD,
    target_fpr: float,
) -> FixedFprCutoff:
    """Choose a validation cutoff by recall, FPR, precision and higher-cutoff order."""
    reference_scores, predicted_scores = _score_pair(
        reference_severity,
        predicted_severity,
    )
    resolved_reference = validate_severity_threshold(
        reference_threshold,
        name="reference threshold",
    )
    resolved_target_fpr = float(target_fpr)
    if not np.isfinite(resolved_target_fpr) or not 0.0 <= resolved_target_fpr <= 1.0:
        raise AlertPolicyInputError("Target FPR must be finite and within [0, 1]")
    reference_labels = severity_labels(reference_scores, threshold=resolved_reference)
    if len(np.unique(reference_labels)) < 2:
        raise AlertPolicyInputError("Fixed-FPR calibration requires both reference classes")

    no_alert_cutoff = float(np.nextafter(np.max(predicted_scores), np.inf))
    candidates = np.unique(np.append(predicted_scores, no_alert_cutoff))
    eligible: list[tuple[tuple[float, float, float, float], FixedFprCutoff]] = []
    for cutoff in candidates:
        predicted_labels = predicted_scores >= cutoff
        tp = int(np.count_nonzero(reference_labels & predicted_labels))
        fp = int(np.count_nonzero(~reference_labels & predicted_labels))
        positive_count = int(np.count_nonzero(reference_labels))
        negative_count = len(reference_labels) - positive_count
        predicted_positive_count = tp + fp
        recall = tp / positive_count
        fpr = fp / negative_count
        precision = tp / predicted_positive_count if predicted_positive_count else 0.0
        if fpr <= resolved_target_fpr + np.finfo(np.float64).eps:
            result = FixedFprCutoff(
                target_fpr=resolved_target_fpr,
                cutoff=float(cutoff),
                validation_recall=recall,
                validation_fpr=fpr,
                validation_precision=precision,
            )
            key = (-recall, fpr, -precision, -float(cutoff))
            eligible.append((key, result))

    if not eligible:  # pragma: no cover - no-alert cutoff always satisfies FPR zero.
        raise RuntimeError("No cutoff satisfies the requested FPR")
    return min(eligible, key=lambda item: item[0])[1]
