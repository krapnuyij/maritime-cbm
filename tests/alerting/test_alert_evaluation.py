from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from maritime_cbm.alerting.benchmark import _write_deterministic_gzip_csv
from maritime_cbm.alerting.evaluation import (
    ModelPredictionBundle,
    build_fixed_fpr_cutoffs,
    build_fixed_fpr_test_metrics,
    build_primary_policy_predictions,
    build_threshold_sensitivity,
)
from maritime_cbm.alerting.reporting import (
    build_primary_alert_metrics,
    build_primary_model_comparison,
    write_alert_figures,
)


def _coefficients(severity: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            1.0 - severity[:, 0] * 0.050,
            1.0 - severity[:, 1] * 0.025,
        )
    )


def _prediction_table(
    model_id: str,
    scenario: str,
    role: str,
    true_severity: np.ndarray,
    predicted_severity: np.ndarray,
) -> pd.DataFrame:
    truth = _coefficients(true_severity)
    predicted = _coefficients(predicted_severity)
    table = pd.DataFrame(
        {
            "model_id": model_id,
            "scenario": scenario,
            "role": role,
            "row_index": np.arange(len(truth)),
            "v": np.resize(np.asarray([3.0, 6.0, 9.0]), len(truth)),
        }
    )
    for index, target in enumerate(("kMc", "kMt")):
        table[f"{target}_true"] = truth[:, index]
        table[f"{target}_predicted"] = predicted[:, index]
        table[f"{target}_error"] = predicted[:, index] - truth[:, index]
    return table


def _bundles() -> tuple[ModelPredictionBundle, ...]:
    validation_true = np.asarray(
        [
            [0.9, 0.1],
            [0.85, 0.1],
            [0.1, 0.9],
            [0.1, 0.85],
            [0.1, 0.1],
            [0.2, 0.2],
        ]
    )
    base_prediction = np.asarray(
        [
            [0.88, 0.1],
            [0.75, 0.1],
            [0.1, 0.88],
            [0.1, 0.75],
            [0.3, 0.2],
            [0.2, 0.3],
        ]
    )
    scenarios = (
        "random_row",
        "state_group",
        "compressor_holdout",
        "turbine_holdout",
    )
    bundles = []
    for model_index, model_id in enumerate(("m2_random_forest", "m3_linear_residual_mlp")):
        predicted = base_prediction + model_index * 0.01
        validation = _prediction_table(
            model_id,
            "state_group",
            "validation",
            validation_true,
            predicted,
        )
        test_tables = []
        for scenario in scenarios:
            scenario_true = validation_true.copy()
            if scenario == "compressor_holdout":
                scenario_true[:, 0] = 0.9
            elif scenario == "turbine_holdout":
                scenario_true[:, 1] = 0.9
            test_tables.append(
                _prediction_table(
                    model_id,
                    scenario,
                    "test",
                    scenario_true,
                    predicted,
                )
            )
        bundles.append(
            ModelPredictionBundle(
                model_id=model_id,
                validation_predictions=validation,
                test_predictions=pd.concat(test_tables, ignore_index=True),
                model_sha256=f"{model_id}-model",
                predictions_sha256=f"{model_id}-predictions",
            )
        )
    return tuple(bundles)


def test_alert_evaluation_tables_preserve_frozen_protocol_counts() -> None:
    bundles = _bundles()

    sensitivity = build_threshold_sensitivity(bundles)
    primary = build_primary_alert_metrics(sensitivity)
    comparison = build_primary_model_comparison(primary)
    cutoffs = build_fixed_fpr_cutoffs(bundles)
    fixed_fpr = build_fixed_fpr_test_metrics(bundles, cutoffs)

    assert len(sensitivity) == 150
    assert len(primary) == 24
    assert len(comparison) == 12
    assert len(cutoffs) == 12
    assert len(fixed_fpr) == 48
    assert set(primary["channel"]) == {"kMc", "kMt", "any"}
    assert set(cutoffs["target_fpr"]) == {0.01, 0.05}


def test_holdout_single_class_metrics_are_explicitly_unavailable() -> None:
    primary = build_primary_alert_metrics(build_threshold_sensitivity(_bundles()))
    compressor = primary.loc[
        (primary["scenario"] == "compressor_holdout") & (primary["channel"] == "kMc")
    ]
    turbine = primary.loc[
        (primary["scenario"] == "turbine_holdout") & (primary["channel"] == "kMt")
    ]

    assert compressor["unavailable_reason"].eq("single_class_all_positive").all()
    assert turbine["unavailable_reason"].eq("single_class_all_positive").all()
    assert compressor["pr_auc"].isna().all()
    assert turbine["fpr"].isna().all()
    assert compressor["recall"].notna().all()


def test_primary_policy_predictions_include_unclipped_severity_and_alerts() -> None:
    table = build_primary_policy_predictions(_bundles())

    assert len(table) == 60
    assert {
        "kMc_true_severity",
        "kMt_predicted_severity",
        "any_reference_alert",
        "any_predicted_alert",
    }.issubset(table.columns)
    assert table["any_reference_alert"].dtype == bool


def test_alert_figures_are_written(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    sensitivity = build_threshold_sensitivity(_bundles())
    primary = build_primary_alert_metrics(sensitivity)

    paths = write_alert_figures(sensitivity, primary, tmp_path)

    assert len(paths) == 2
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_row_predictions_gzip_is_deterministic_across_paths(tmp_path: Path) -> None:
    frame = build_primary_policy_predictions(_bundles())
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"

    _write_deterministic_gzip_csv(frame, first)
    _write_deterministic_gzip_csv(frame, second)

    assert first.read_bytes() == second.read_bytes()
    pd.testing.assert_frame_equal(pd.read_csv(first), pd.read_csv(second))
