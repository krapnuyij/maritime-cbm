import numpy as np
import pytest

from maritime_cbm.alerting.policy import (
    AlertPolicyInputError,
    channel_severity,
    coefficients_to_severity,
    evaluate_alert_policy,
    severity_states,
)


def test_coefficients_map_to_unclipped_component_severity() -> None:
    coefficients = np.asarray(
        [
            [1.000, 1.000],
            [0.975, 0.9875],
            [0.960, 0.980],
            [0.940, 0.970],
        ]
    )

    severity = coefficients_to_severity(coefficients)

    assert severity == pytest.approx(
        np.asarray(
            [
                [0.0, 0.0],
                [0.5, 0.5],
                [0.8, 0.8],
                [1.2, 1.2],
            ]
        )
    )
    assert np.max(severity) > 1.0


def test_policy_boundaries_are_inclusive_and_overall_uses_worst_component() -> None:
    coefficients = np.asarray(
        [
            [1.000, 1.000],
            [0.975, 1.000],
            [1.000, 0.980],
        ]
    )

    decision = evaluate_alert_policy(coefficients)

    assert decision.compressor_state.tolist() == ["normal", "watch", "normal"]
    assert decision.turbine_state.tolist() == ["normal", "normal", "alert"]
    assert decision.overall_state.tolist() == ["normal", "watch", "alert"]
    assert channel_severity(decision.severity, "any") == pytest.approx([0.0, 0.5, 0.8])


def test_policy_rejects_invalid_shape_and_threshold_order() -> None:
    with pytest.raises(AlertPolicyInputError, match="shape"):
        coefficients_to_severity(np.asarray([0.96, 0.98]))

    with pytest.raises(AlertPolicyInputError, match="lower"):
        severity_states(np.asarray([0.5]), watch_threshold=0.8, alert_threshold=0.8)
