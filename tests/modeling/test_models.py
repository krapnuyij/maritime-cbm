import numpy as np
import pandas as pd

from maritime_cbm.data.schema import MODEL_FEATURE_COLUMNS
from maritime_cbm.modeling.models import (
    RANDOM_FOREST_ESTIMATORS,
    build_candidate_specs,
    build_estimator,
)


def _training_data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(42)
    speeds = np.repeat([3.0, 6.0, 9.0], 12)
    values = rng.normal(size=(len(speeds), len(MODEL_FEATURE_COLUMNS)))
    frame = pd.DataFrame(values, columns=MODEL_FEATURE_COLUMNS)
    frame.loc[:, "v"] = speeds
    targets = np.column_stack(
        (
            0.975 + 0.005 * values[:, 1],
            0.9875 + 0.0025 * values[:, 2],
        )
    )
    return frame, targets


def test_candidate_grid_is_complete_and_deterministic() -> None:
    first = build_candidate_specs()
    second = build_candidate_specs()

    assert first == second
    assert len(first) == 17
    assert len({spec.candidate_id for spec in first}) == 17
    assert sum(spec.family == "dummy" for spec in first) == 1
    assert sum(spec.family == "ridge" for spec in first) == 10
    assert sum(spec.family == "random_forest" for spec in first) == 6
    assert all(
        spec.parameter_dict()["n_estimators"] == RANDOM_FOREST_ESTIMATORS
        for spec in first
        if spec.family == "random_forest"
    )
    assert not first[0].selectable


def test_each_model_family_supports_two_output_prediction() -> None:
    features, targets = _training_data()
    candidates = build_candidate_specs()
    representatives = (
        next(spec for spec in candidates if spec.family == "dummy"),
        next(spec for spec in candidates if spec.candidate_id.startswith("ridge_raw")),
        next(spec for spec in candidates if spec.candidate_id.startswith("ridge_speed_centered")),
        next(spec for spec in candidates if spec.family == "random_forest"),
    )

    for spec in representatives:
        predictions = build_estimator(spec, seed=42).fit(features, targets).predict(features)
        assert predictions.shape == targets.shape
        assert np.isfinite(predictions).all()


def test_random_forest_predictions_are_seed_reproducible() -> None:
    features, targets = _training_data()
    spec = next(spec for spec in build_candidate_specs() if spec.family == "random_forest")

    first = build_estimator(spec, seed=42).fit(features, targets).predict(features)
    second = build_estimator(spec, seed=42).fit(features, targets).predict(features)

    assert np.array_equal(first, second)
