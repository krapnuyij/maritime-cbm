from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

import maritime_cbm.modeling.torch_training as torch_training
from maritime_cbm.data.schema import MODEL_FEATURE_COLUMNS
from maritime_cbm.modeling.preprocessing import ModelingInputError
from maritime_cbm.modeling.torch_models import build_torch_candidate_specs
from maritime_cbm.modeling.torch_training import (
    EarlyStopping,
    FittedTorchPreprocessor,
    TrainingConfig,
    fit_torch_regressor,
    load_torch_checkpoint,
    resolve_torch_device,
    save_torch_checkpoint,
)


def _regression_data(row_count: int = 48) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(42)
    features = pd.DataFrame(
        rng.normal(size=(row_count, len(MODEL_FEATURE_COLUMNS))),
        columns=MODEL_FEATURE_COLUMNS,
    )
    features.loc[:, "v"] = np.resize(np.asarray([3.0, 6.0, 9.0]), row_count)
    kmc = 0.975 + 0.006 * np.tanh(features["T2"].to_numpy())
    kmt = 0.9875 + 0.003 * np.tanh(features["P2"].to_numpy())
    return features, np.column_stack((kmc, kmt))


def _short_config() -> TrainingConfig:
    return TrainingConfig(
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=8,
        max_epochs=8,
        min_epochs=4,
        patience=3,
        min_delta=1e-5,
    )


def test_preprocessor_fits_train_only_and_round_trips_payload() -> None:
    features, targets = _regression_data()
    train_features = features.iloc[:36]
    train_targets = targets[:36]
    validation_features = features.iloc[36:].copy()
    validation_features.loc[:, "T2"] += 10.0
    validation_targets = targets[36:] + np.asarray([0.01, 0.005])

    preprocessor = FittedTorchPreprocessor.fit(
        train_features,
        train_targets,
        preprocessing="speed_centered",
    )
    restored = FittedTorchPreprocessor.from_payload(preprocessor.to_payload())

    assert np.mean(preprocessor.transform_features(train_features), axis=0) == pytest.approx(
        np.zeros(len(MODEL_FEATURE_COLUMNS)), abs=1e-6
    )
    assert not np.allclose(np.mean(preprocessor.transform_features(validation_features), axis=0), 0)
    assert not np.allclose(preprocessor.transform_targets(validation_targets).mean(axis=0), 0)
    assert restored.transform_features(validation_features) == pytest.approx(
        preprocessor.transform_features(validation_features)
    )

    raw_preprocessor = FittedTorchPreprocessor.fit(
        train_features,
        train_targets,
        preprocessing="raw",
    )
    restored_raw = FittedTorchPreprocessor.from_payload(raw_preprocessor.to_payload())
    assert restored_raw.transform_features(validation_features) == pytest.approx(
        raw_preprocessor.transform_features(validation_features)
    )


def test_speed_centering_rejects_unseen_speed() -> None:
    features, targets = _regression_data()
    preprocessor = FittedTorchPreprocessor.fit(
        features,
        targets,
        preprocessing="speed_centered",
    )
    unseen = features.iloc[:1].copy()
    unseen.loc[:, "v"] = 12.0

    with pytest.raises(ModelingInputError, match="No training statistics"):
        preprocessor.transform_features(unseen)


def test_early_stopping_restores_the_best_synthetic_checkpoint() -> None:
    config = TrainingConfig(
        batch_size=1,
        max_epochs=10,
        min_epochs=1,
        patience=2,
        min_delta=0.01,
    )
    model = nn.Linear(1, 1, bias=False)
    stopper = EarlyStopping(config)
    stop_flags = []

    for epoch, score in enumerate((1.0, 0.5, 0.51, 0.52), start=1):
        with torch.no_grad():
            model.weight.fill_(float(epoch))
        stop_flags.append(stopper.update(epoch, score, model))

    assert stop_flags == [False, False, False, True]
    assert stopper.best_epoch == 2
    stopper.restore(model)
    assert model.weight.item() == pytest.approx(2.0)


def test_cpu_training_is_deterministic_for_the_same_seed() -> None:
    features, targets = _regression_data()
    candidate = build_torch_candidate_specs()[0]

    first = fit_torch_regressor(
        features.iloc[:36],
        targets[:36],
        features.iloc[36:],
        targets[36:],
        candidate=candidate,
        seed=42,
        config=_short_config(),
    )
    second = fit_torch_regressor(
        features.iloc[:36],
        targets[:36],
        features.iloc[36:],
        targets[36:],
        candidate=candidate,
        seed=42,
        config=_short_config(),
    )

    first_predictions = first.regressor.predict(features.iloc[36:])
    second_predictions = second.regressor.predict(features.iloc[36:])
    assert np.array_equal(first_predictions, second_predictions)
    assert first.best_epoch == second.best_epoch
    assert first.history == second.history


def test_checkpoint_round_trip_preserves_predictions(tmp_path: Path) -> None:
    features, targets = _regression_data()
    candidate = build_torch_candidate_specs()[2]
    result = fit_torch_regressor(
        features.iloc[:36],
        targets[:36],
        features.iloc[36:],
        targets[36:],
        candidate=candidate,
        seed=42,
        config=_short_config(),
    )
    checkpoint_path = tmp_path / "model.pt"

    save_torch_checkpoint(result, checkpoint_path, metadata={"scenario": "state_group"})
    loaded = load_torch_checkpoint(checkpoint_path)

    expected = result.regressor.predict(features.iloc[36:])
    actual = loaded.regressor.predict(features.iloc[36:])
    assert np.array_equal(actual, expected)
    assert loaded.metadata == {"scenario": "state_group"}
    assert loaded.trainable_parameter_count == result.trainable_parameter_count
    assert loaded.torch_version == str(torch.__version__)
    assert loaded.training_seconds == pytest.approx(result.training_seconds)


def test_checkpoint_rejects_non_primitive_metadata(tmp_path: Path) -> None:
    features, targets = _regression_data()
    result = fit_torch_regressor(
        features.iloc[:36],
        targets[:36],
        features.iloc[36:],
        targets[36:],
        candidate=build_torch_candidate_specs()[0],
        seed=42,
        config=_short_config(),
    )

    with pytest.raises(TypeError, match="unsupported type ndarray"):
        save_torch_checkpoint(
            result,
            tmp_path / "unsafe.pt",
            metadata={"array": np.asarray([1.0])},
        )


def test_mps_preference_falls_back_when_backend_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    resolution = resolve_torch_device("mps")

    assert resolution.resolved == "cpu"
    assert resolution.fallback_reason == "MPS is not available; using CPU"


def test_mps_runtime_error_restarts_training_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    features, targets = _regression_data()
    candidate = build_torch_candidate_specs()[0]
    original_run_training = torch_training._run_training

    def fail_only_on_mps(*args: object, **kwargs: object) -> object:
        if kwargs["device_name"] == "mps":
            raise RuntimeError("deterministic operation is unavailable on MPS")
        return original_run_training(*args, **kwargs)

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch_training, "_run_training", fail_only_on_mps)

    result = fit_torch_regressor(
        features.iloc[:36],
        targets[:36],
        features.iloc[36:],
        targets[36:],
        candidate=candidate,
        seed=42,
        config=_short_config(),
        device="mps",
    )

    assert result.requested_device == "mps"
    assert result.resolved_device == "cpu"
    assert result.fallback_reason is not None
    assert "MPS training failed" in result.fallback_reason
