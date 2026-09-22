from pathlib import Path

import numpy as np
import pandas as pd

import maritime_cbm.modeling.torch_evaluation as torch_evaluation
from maritime_cbm.data.schema import ALL_COLUMNS, FEATURE_COLUMNS
from maritime_cbm.data.splitting import DatasetSplit
from maritime_cbm.modeling.torch_evaluation import (
    FINAL_EVALUATION_SCENARIOS,
    SELECTION_SCENARIOS,
    run_torch_final_evaluation,
    run_torch_model_selection,
)
from maritime_cbm.modeling.torch_models import build_torch_candidate_specs
from maritime_cbm.modeling.torch_training import (
    TrainingConfig,
    load_torch_checkpoint,
    save_torch_checkpoint,
)


def _small_frame(row_count: int = 48) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        rng.normal(size=(row_count, len(ALL_COLUMNS))),
        columns=ALL_COLUMNS,
    )
    speeds = np.resize(np.asarray([3.0, 6.0, 9.0]), row_count)
    frame.loc[:, "v"] = speeds
    frame.loc[:, "lp"] = speeds
    frame.loc[:, "kMc"] = 0.975 + 0.006 * np.tanh(frame["T2"].to_numpy())
    frame.loc[:, "kMt"] = 0.9875 + 0.003 * np.tanh(frame["P2"].to_numpy())
    assert set(FEATURE_COLUMNS).issubset(frame.columns)
    return frame


def _small_splits() -> dict[str, DatasetSplit]:
    return {
        scenario: DatasetSplit(
            name=scenario,
            train=np.arange(0, 30, dtype=np.int64),
            validation=np.arange(30, 39, dtype=np.int64),
            test=np.arange(39, 48, dtype=np.int64),
        )
        for scenario in FINAL_EVALUATION_SCENARIOS
    }


def _short_config() -> TrainingConfig:
    return TrainingConfig(
        batch_size=8,
        max_epochs=4,
        min_epochs=2,
        patience=2,
        min_delta=1e-5,
    )


def test_selection_uses_only_validation_and_repeated_seeds(
    monkeypatch,
) -> None:
    frame = _small_frame()
    splits = _small_splits()
    monkeypatch.setattr(torch_evaluation, "build_dataset_splits", lambda frame, seed: splits)
    candidates = (build_torch_candidate_specs()[0],)

    outcome = run_torch_model_selection(
        frame,
        base_seed=42,
        candidates=candidates,
        selection_seeds=(42, 43),
        config=_short_config(),
    )

    assert outcome.selected_candidate == candidates[0]
    assert len(outcome.evaluations) == len(SELECTION_SCENARIOS) * 2
    assert {result.scenario for result in outcome.evaluations} == set(SELECTION_SCENARIOS)
    assert {result.seed for result in outcome.evaluations} == {42, 43}
    assert all(len(result.validation_row_indices) == 9 for result in outcome.evaluations)


def test_final_evaluation_reuses_selection_checkpoints_and_fits_random_row(
    monkeypatch,
    tmp_path: Path,
) -> None:
    frame = _small_frame()
    splits = _small_splits()
    monkeypatch.setattr(torch_evaluation, "build_dataset_splits", lambda frame, seed: splits)
    candidate = build_torch_candidate_specs()[0]
    config = _short_config()
    selection = run_torch_model_selection(
        frame,
        base_seed=42,
        candidates=(candidate,),
        selection_seeds=(42,),
        config=config,
    )
    checkpoints = {}
    for scenario in SELECTION_SCENARIOS:
        result = selection.selected_evaluation(scenario, 42).training_result
        path = tmp_path / f"{scenario}.pt"
        save_torch_checkpoint(result, path, metadata={"scenario": scenario})
        checkpoints[scenario] = load_torch_checkpoint(path)

    outcome = run_torch_final_evaluation(
        frame,
        candidate,
        checkpoints,
        seed=42,
        config=config,
    )

    assert {result.scenario for result in outcome.evaluations} == set(FINAL_EVALUATION_SCENARIOS)
    assert (
        next(
            result for result in outcome.evaluations if result.scenario == "random_row"
        ).training_source
        == "final_fit"
    )
    assert all(
        result.training_source == "selection_checkpoint"
        for result in outcome.evaluations
        if result.scenario != "random_row"
    )
    assert all(len(result.role("test").predictions) == 9 for result in outcome.evaluations)
