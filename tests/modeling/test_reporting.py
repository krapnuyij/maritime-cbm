from pathlib import Path

import numpy as np
import pandas as pd

from maritime_cbm.data.schema import ALL_COLUMNS
from maritime_cbm.data.splitting import DatasetSplit
from maritime_cbm.modeling.evaluation import (
    ScenarioEvaluation,
    SelectionOutcome,
    _fit_and_evaluate,
)
from maritime_cbm.modeling.models import build_candidate_specs
from maritime_cbm.modeling.reporting import (
    build_error_by_speed,
    build_error_by_target_state,
    build_metric_table,
    build_prediction_table,
    build_selection_summary,
    write_model_figures,
)


def _small_evaluation() -> tuple[pd.DataFrame, tuple[ScenarioEvaluation, ...]]:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(18, len(ALL_COLUMNS))), columns=ALL_COLUMNS)
    frame.loc[:, "v"] = np.repeat([3.0, 6.0, 9.0], 6)
    frame.loc[:, "kMc"] = np.linspace(0.95, 1.0, len(frame))
    frame.loc[:, "kMt"] = np.linspace(0.975, 1.0, len(frame))
    split = DatasetSplit(
        name="state_group",
        train=np.arange(0, 10),
        validation=np.arange(10, 14),
        test=np.arange(14, 18),
    )
    candidate = next(spec for spec in build_candidate_specs() if spec.selectable)
    evaluation = _fit_and_evaluate(
        frame,
        split,
        candidate,
        roles=("validation", "test"),
        seed=42,
    )
    return frame, (evaluation,)


def test_reporting_tables_preserve_scenarios_targets_and_rows() -> None:
    frame, evaluations = _small_evaluation()

    metrics = build_metric_table(evaluations)
    predictions = build_prediction_table(frame, evaluations)
    by_speed = build_error_by_speed(predictions)
    by_state = build_error_by_target_state(predictions)

    assert set(metrics["target"]) == {"kMc", "kMt"}
    assert set(metrics["role"]) == {"validation", "test"}
    assert predictions["row_index"].tolist() == [14, 15, 16, 17]
    assert set(by_speed["target"]) == {"kMc", "kMt"}
    assert set(by_state["target"]) == {"kMc", "kMt"}


def test_model_figures_are_optional_and_written_with_matplotlib(tmp_path: Path) -> None:
    frame, evaluations = _small_evaluation()
    predictions = build_prediction_table(frame, evaluations)
    predictions = pd.concat(
        [
            predictions,
            predictions.assign(scenario="compressor_holdout"),
            predictions.assign(scenario="turbine_holdout"),
        ],
        ignore_index=True,
    )

    paths = write_model_figures(predictions, tmp_path)

    assert len(paths) == 3
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_selection_summary_uses_state_group_metrics_and_marks_selected() -> None:
    _, evaluations = _small_evaluation()
    evaluation = evaluations[0]
    outcome = SelectionOutcome(
        selected_candidate=evaluation.candidate,
        selection_key=(0.1, 0.2, 0, evaluation.candidate.candidate_id),
        evaluations=evaluations,
        split_hashes={},
    )

    summary = build_selection_summary(outcome)

    assert len(summary) == 1
    assert bool(summary.loc[0, "selected"])
    assert summary.loc[0, "candidate_id"] == evaluation.candidate.candidate_id
