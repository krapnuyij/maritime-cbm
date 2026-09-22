import numpy as np
import pytest

from maritime_cbm.modeling.metrics import (
    MetricInputError,
    compute_regression_metrics,
    state_group_selection_key,
)


def test_regression_metrics_are_target_specific_and_unclipped() -> None:
    y_true = np.asarray([[0.95, 0.975], [1.0, 1.0]])
    y_pred = np.asarray([[0.94, 0.98], [1.01, 0.99]])

    metrics = compute_regression_metrics(y_true, y_pred)
    by_target = {metric.target: metric for metric in metrics}

    assert by_target["kMc"].mae == pytest.approx(0.01)
    assert by_target["kMc"].rmse == pytest.approx(0.01)
    assert by_target["kMc"].nrmse == pytest.approx(0.2)
    assert by_target["kMc"].below_range_count == 1
    assert by_target["kMc"].above_range_count == 1
    assert by_target["kMt"].mae == pytest.approx(0.0075)
    assert by_target["kMt"].nrmse == pytest.approx(np.sqrt((0.005**2 + 0.01**2) / 2) / 0.025)


def test_selection_key_uses_mean_then_worst_target_then_complexity() -> None:
    balanced = compute_regression_metrics(
        np.asarray([[0.95, 0.975], [1.0, 1.0]]),
        np.asarray([[0.955, 0.9775], [0.995, 0.9975]]),
    )
    unbalanced = compute_regression_metrics(
        np.asarray([[0.95, 0.975], [1.0, 1.0]]),
        np.asarray([[0.95, 0.98], [1.0, 0.995]]),
    )

    balanced_key = state_group_selection_key(balanced, complexity_rank=3, candidate_id="balanced")
    unbalanced_key = state_group_selection_key(
        unbalanced, complexity_rank=1, candidate_id="unbalanced"
    )

    assert balanced_key[0] == pytest.approx(unbalanced_key[0])
    assert balanced_key[1] < unbalanced_key[1]
    assert balanced_key < unbalanced_key


def test_metrics_reject_wrong_shape_or_non_finite_values() -> None:
    with pytest.raises(MetricInputError, match="shape"):
        compute_regression_metrics(np.ones((3, 1)), np.ones((3, 1)))

    with pytest.raises(MetricInputError, match="finite"):
        compute_regression_metrics(np.ones((3, 2)), np.asarray([[1.0, np.nan]] * 3))
