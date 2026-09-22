import numpy as np
import pytest

from maritime_cbm.modeling.metrics import (
    MetricInputError,
    TargetRegressionMetrics,
    aggregated_state_group_selection_key,
    compute_regression_metrics,
    state_group_selection_key,
)


def _metrics_with_nrmse(kmc: float, kmt: float) -> tuple[TargetRegressionMetrics, ...]:
    return tuple(
        TargetRegressionMetrics(
            target=target,
            sample_count=10,
            mae=0.0,
            rmse=0.0,
            r2=0.0,
            bias=0.0,
            absolute_error_p95=0.0,
            below_range_count=0,
            above_range_count=0,
            nrmse=nrmse,
        )
        for target, nrmse in (("kMc", kmc), ("kMt", kmt))
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


def test_aggregate_selection_key_averages_each_target_before_taking_worst() -> None:
    runs = (
        _metrics_with_nrmse(0.035, 0.024),
        _metrics_with_nrmse(0.034, 0.023),
        _metrics_with_nrmse(0.010, 0.011),
    )

    key = aggregated_state_group_selection_key(
        runs,
        trainable_parameter_count=2_978,
        architecture_rank=1,
        candidate_id="mlp_raw_hidden_64_32",
    )

    assert key[0] == pytest.approx(0.022833333333333334)
    assert key[1] == pytest.approx(0.026333333333333334)
    assert np.mean([max(metric.nrmse for metric in run) for run in runs]) == pytest.approx(
        0.02666666666666667
    )
    assert key[2:] == (2_978, 1, "mlp_raw_hidden_64_32")


def test_aggregate_selection_key_rejects_empty_or_incomplete_runs() -> None:
    with pytest.raises(MetricInputError, match="At least one"):
        aggregated_state_group_selection_key(
            (),
            trainable_parameter_count=1,
            architecture_rank=1,
            candidate_id="candidate",
        )

    with pytest.raises(MetricInputError, match="Expected one metric"):
        aggregated_state_group_selection_key(
            (_metrics_with_nrmse(0.01, 0.02)[:1],),
            trainable_parameter_count=1,
            architecture_rank=1,
            candidate_id="candidate",
        )
