from itertools import product

import numpy as np
import pandas as pd

from maritime_cbm.data.schema import (
    EXPECTED_KMC_VALUES,
    EXPECTED_KMT_VALUES,
    FEATURE_COLUMNS,
)
from maritime_cbm.modeling.evaluation import run_final_evaluation, run_model_selection
from maritime_cbm.modeling.models import build_candidate_specs


def _release_like_frame() -> pd.DataFrame:
    rows = []
    for kmc, kmt, speed in product(
        EXPECTED_KMC_VALUES,
        EXPECTED_KMT_VALUES,
        range(3, 28, 3),
    ):
        row = {column: float(speed) for column in FEATURE_COLUMNS}
        for index, column in enumerate(FEATURE_COLUMNS):
            row[column] += index * 0.1 + (1.0 - kmc) * 10 + (1.0 - kmt) * 20
        row["v"] = float(speed)
        row["kMc"] = kmc
        row["kMt"] = kmt
        rows.append(row)
    return pd.DataFrame(rows)


def test_selection_evaluates_validation_roles_without_test_access() -> None:
    frame = _release_like_frame()
    specs = build_candidate_specs()
    candidates = (
        specs[0],
        next(spec for spec in specs if spec.candidate_id.startswith("ridge_raw")),
    )

    outcome = run_model_selection(frame, seed=42, candidates=candidates)

    assert outcome.selected_candidate.selectable
    assert {evaluation.scenario for evaluation in outcome.evaluations} == {
        "state_group",
        "compressor_holdout",
        "turbine_holdout",
    }
    assert all(
        tuple(role.role for role in evaluation.roles) == ("validation",)
        for evaluation in outcome.evaluations
    )


def test_final_evaluation_uses_all_scenarios_and_preserves_test_rows() -> None:
    frame = _release_like_frame()
    candidate = build_candidate_specs()[0]

    outcome = run_final_evaluation(frame, candidate, seed=42)

    assert {evaluation.scenario for evaluation in outcome.evaluations} == {
        "random_row",
        "state_group",
        "compressor_holdout",
        "turbine_holdout",
    }
    assert all(
        tuple(role.role for role in evaluation.roles) == ("validation", "test")
        for evaluation in outcome.evaluations
    )
    assert all(
        len(evaluation.role("test").predictions) == len(evaluation.role("test").row_indices)
        for evaluation in outcome.evaluations
    )
    assert np.isfinite(outcome.evaluations[0].role("test").predictions).all()
