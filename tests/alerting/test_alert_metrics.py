import numpy as np
import pytest

import maritime_cbm.alerting.metrics as alert_metrics
from maritime_cbm.alerting.metrics import (
    compute_alert_metrics,
    select_fixed_fpr_cutoff,
)
from maritime_cbm.alerting.policy import AlertPolicyInputError


def test_two_class_metrics_include_pr_auc_and_confusion_counts() -> None:
    reference = np.asarray([0.9, 0.85, 0.7, 0.2])
    predicted = np.asarray([0.95, 0.75, 0.9, 0.1])

    metrics = compute_alert_metrics(reference, predicted)

    assert (metrics.tp, metrics.fn, metrics.fp, metrics.tn) == (1, 1, 1, 1)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)
    assert metrics.fpr == pytest.approx(0.5)
    assert metrics.miss_rate == pytest.approx(0.5)
    assert metrics.pr_auc is not None
    assert metrics.metric_valid
    assert metrics.unavailable_reason is None


def test_all_positive_reference_skips_pr_auc_and_discrimination_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> float:
        raise AssertionError("average_precision_score must not run for one class")

    monkeypatch.setattr(alert_metrics, "average_precision_score", fail_if_called)

    metrics = compute_alert_metrics(
        np.asarray([0.8, 0.9, 1.0]),
        np.asarray([0.7, 0.85, 1.1]),
    )

    assert (metrics.tp, metrics.fn) == (2, 1)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.miss_rate == pytest.approx(1 / 3)
    assert metrics.precision is None
    assert metrics.f1 is None
    assert metrics.fpr is None
    assert metrics.pr_auc is None
    assert not metrics.metric_valid
    assert metrics.unavailable_reason == "single_class_all_positive"


def test_all_negative_reference_reports_only_false_positive_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> float:
        raise AssertionError("average_precision_score must not run for one class")

    monkeypatch.setattr(alert_metrics, "average_precision_score", fail_if_called)

    metrics = compute_alert_metrics(
        np.asarray([0.1, 0.2, 0.3]),
        np.asarray([0.9, 0.2, 0.3]),
    )

    assert (metrics.fp, metrics.tn) == (1, 2)
    assert metrics.fpr == pytest.approx(1 / 3)
    assert metrics.precision is None
    assert metrics.recall is None
    assert metrics.f1 is None
    assert metrics.miss_rate is None
    assert metrics.pr_auc is None
    assert not metrics.metric_valid
    assert metrics.unavailable_reason == "single_class_all_negative"


def test_fixed_fpr_cutoff_maximizes_recall_under_validation_constraint() -> None:
    reference = np.asarray([0.9, 0.85, 0.82, 0.7, 0.6, 0.5])
    predicted = np.asarray([0.95, 0.75, 0.70, 0.80, 0.40, 0.30])

    cutoff = select_fixed_fpr_cutoff(
        reference,
        predicted,
        target_fpr=0.0,
    )

    assert cutoff.cutoff == pytest.approx(0.95)
    assert cutoff.validation_recall == pytest.approx(1 / 3)
    assert cutoff.validation_fpr == 0.0
    assert cutoff.validation_precision == 1.0


def test_fixed_fpr_cutoff_rejects_single_class_reference() -> None:
    with pytest.raises(AlertPolicyInputError, match="both reference classes"):
        select_fixed_fpr_cutoff(
            np.asarray([0.8, 0.9]),
            np.asarray([0.7, 0.8]),
            target_fpr=0.05,
        )


def test_metrics_allow_unclipped_fixed_fpr_decision_cutoff() -> None:
    metrics = compute_alert_metrics(
        np.asarray([0.9, 0.1]),
        np.asarray([1.2, 0.2]),
        decision_threshold=1.1,
    )

    assert metrics.tp == 1
    assert metrics.tn == 1
    assert metrics.decision_threshold == 1.1
