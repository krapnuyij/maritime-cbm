from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from maritime_cbm.modeling.torch_reporting import (
    build_baseline_comparison,
    build_holdout_diagnostics,
    write_torch_figures,
)


def _metric_table(scale: float) -> pd.DataFrame:
    rows = []
    for scenario in (
        "random_row",
        "state_group",
        "compressor_holdout",
        "turbine_holdout",
    ):
        for target in ("kMc", "kMt"):
            rows.append(
                {
                    "scenario": scenario,
                    "role": "test",
                    "target": target,
                    "mae": 0.1 * scale,
                    "rmse": 0.2 * scale,
                    "r2": 0.3 * scale,
                    "bias": 0.01 * scale,
                    "absolute_error_p95": 0.4 * scale,
                    "nrmse": 0.5 * scale,
                }
            )
    return pd.DataFrame(rows)


def _predictions() -> pd.DataFrame:
    rows = []
    for scenario, target, truth in (
        ("compressor_holdout", "kMc", np.linspace(0.950, 0.954, 10)),
        ("turbine_holdout", "kMt", np.linspace(0.975, 0.979, 10)),
    ):
        for value in truth:
            rows.append(
                {
                    "scenario": scenario,
                    "kMc_true": value if target == "kMc" else 0.98,
                    "kMc_predicted": value + 0.001 if target == "kMc" else 0.981,
                    "kMt_true": value if target == "kMt" else 0.99,
                    "kMt_predicted": value + 0.001 if target == "kMt" else 0.991,
                }
            )
    return pd.DataFrame(rows)


def test_baseline_comparison_computes_m3_deltas() -> None:
    comparison = build_baseline_comparison(_metric_table(1.0), _metric_table(0.5))

    assert len(comparison) == 8
    assert comparison["nrmse_delta"].to_numpy() == pytest.approx(-0.25)
    assert comparison["nrmse_relative_change"].to_numpy() == pytest.approx(-0.5)


def test_holdout_diagnostics_quantify_linear_extrapolation() -> None:
    diagnostics = build_holdout_diagnostics(_predictions())

    assert set(diagnostics["target"]) == {"kMc", "kMt"}
    assert diagnostics["prediction_slope"].to_numpy() == pytest.approx(1.0)
    assert diagnostics["bias"].to_numpy() == pytest.approx(0.001)


def test_torch_figures_are_written(tmp_path: Path) -> None:
    history = pd.DataFrame(
        {
            "scenario": ["state_group"] * 3,
            "epoch": [1, 2, 3],
            "train_loss": [1.0, 0.5, 0.25],
            "validation_mean_nrmse": [0.3, 0.2, 0.1],
            "best_epoch": [3, 3, 3],
        }
    )

    paths = write_torch_figures(
        history,
        build_baseline_comparison(_metric_table(1.0), _metric_table(0.5)),
        _predictions(),
        tmp_path,
    )

    assert len(paths) == 3
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
