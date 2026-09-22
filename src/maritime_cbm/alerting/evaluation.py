"""Verified M2/M3 predictions and fixed downstream alert-policy evaluation."""

import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch

from maritime_cbm.alerting.metrics import (
    AlertClassificationMetrics,
    FixedFprCutoff,
    compute_alert_metrics,
    select_fixed_fpr_cutoff,
)
from maritime_cbm.alerting.policy import (
    ALERT_CHANNELS,
    ALERT_SEVERITY_THRESHOLD,
    SENSITIVITY_THRESHOLDS,
    AlertChannel,
    channel_severity,
    coefficients_to_severity,
    parse_alert_channel,
)
from maritime_cbm.data.features import select_model_features
from maritime_cbm.data.loader import compute_sha256
from maritime_cbm.data.schema import TARGET_COLUMNS
from maritime_cbm.data.splitting import DatasetSplit, SplitMapping, compute_split_hashes
from maritime_cbm.modeling.torch_training import load_torch_checkpoint

MODEL_IDS: tuple[str, ...] = ("m2_random_forest", "m3_linear_residual_mlp")
FIXED_FPR_TARGETS: tuple[float, ...] = (0.01, 0.05)
PREDICTION_REQUIRED_COLUMNS: tuple[str, ...] = (
    "scenario",
    "role",
    "row_index",
    "v",
    "kMc_true",
    "kMc_predicted",
    "kMc_error",
    "kMt_true",
    "kMt_predicted",
    "kMt_error",
)


class AlertArtifactError(ValueError):
    """Raised when an upstream M2/M3 artifact cannot be trusted or aligned."""


class Predictor(Protocol):
    """Structural protocol for trusted fitted M2/M3 regression bundles."""

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return two-target predictions."""


@dataclass(frozen=True, slots=True)
class ModelPredictionBundle:
    """Verified state-group validation and all-scenario test predictions."""

    model_id: str
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    model_sha256: str
    predictions_sha256: str


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise AlertArtifactError(f"Cannot load upstream evaluation manifest: {path}") from error
    if not isinstance(payload, dict):
        raise AlertArtifactError("Upstream evaluation manifest must contain a JSON object")
    return payload


def _current_m2_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def _verify_hash(path: Path, expected: object, *, label: str) -> str:
    if not isinstance(expected, str):
        raise AlertArtifactError(f"{label} manifest SHA-256 is missing")
    try:
        actual = compute_sha256(path)
    except FileNotFoundError as error:
        raise AlertArtifactError(f"{label} artifact is missing: {path}") from error
    if actual != expected:
        raise AlertArtifactError(f"{label} artifact SHA-256 differs")
    return actual


def _verify_manifest_common(
    payload: dict[str, object],
    *,
    split_hashes: dict[str, dict[str, str]],
    expected_versions: dict[str, str],
    label: str,
) -> None:
    if payload.get("split_hashes") != split_hashes:
        raise AlertArtifactError(f"{label} split hashes differ from the current release")
    if payload.get("runtime_versions") != expected_versions:
        raise AlertArtifactError(f"{label} runtime versions differ from the current environment")


def _prediction_values(table: pd.DataFrame, suffix: str) -> np.ndarray:
    return table.loc[:, [f"{target}_{suffix}" for target in TARGET_COLUMNS]].to_numpy(
        dtype=np.float64,
        copy=True,
    )


def validate_prediction_table(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    splits: SplitMapping,
) -> None:
    """Validate stored test predictions against the current rows and split roles."""
    missing = tuple(
        column for column in PREDICTION_REQUIRED_COLUMNS if column not in predictions.columns
    )
    if missing:
        raise AlertArtifactError(f"Prediction artifact columns are missing: {missing}")
    if set(predictions["scenario"]) != set(splits):
        raise AlertArtifactError("Prediction artifact scenarios differ from approved splits")
    if set(predictions["role"]) != {"test"}:
        raise AlertArtifactError("Prediction artifact must contain test roles only")
    if predictions.duplicated(["scenario", "row_index"]).any():
        raise AlertArtifactError("Prediction artifact contains duplicate scenario rows")

    numeric_columns = [
        "row_index",
        "v",
        "kMc_true",
        "kMc_predicted",
        "kMc_error",
        "kMt_true",
        "kMt_predicted",
        "kMt_error",
    ]
    if not np.isfinite(predictions.loc[:, numeric_columns].to_numpy(dtype=np.float64)).all():
        raise AlertArtifactError("Prediction artifact must contain finite numeric values")
    raw_row_indices = predictions["row_index"].to_numpy(dtype=np.float64)
    if not np.array_equal(raw_row_indices, np.rint(raw_row_indices)):
        raise AlertArtifactError("Prediction row indices must be integers")

    for scenario, split in splits.items():
        subset = predictions.loc[predictions["scenario"] == scenario].sort_values("row_index")
        row_indices = subset["row_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(row_indices, split.test):
            raise AlertArtifactError(f"Prediction rows differ from {scenario} test indices")
        source = frame.iloc[split.test]
        expected_true = source.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float64)
        expected_speed = source["v"].to_numpy(dtype=np.float64)
        true_values = _prediction_values(subset, "true")
        predicted_values = _prediction_values(subset, "predicted")
        error_values = _prediction_values(subset, "error")
        if not np.allclose(true_values, expected_true, rtol=0.0, atol=1e-12):
            raise AlertArtifactError(f"Prediction truths differ for {scenario}")
        if not np.allclose(
            subset["v"].to_numpy(dtype=np.float64),
            expected_speed,
            rtol=0.0,
            atol=1e-12,
        ):
            raise AlertArtifactError(f"Prediction speeds differ for {scenario}")
        if not np.allclose(error_values, predicted_values - true_values, rtol=0.0, atol=1e-12):
            raise AlertArtifactError(f"Prediction errors differ for {scenario}")


def _build_prediction_table(
    frame: pd.DataFrame,
    split: DatasetSplit,
    predictor: Predictor,
    *,
    model_id: str,
) -> pd.DataFrame:
    features = select_model_features(frame).iloc[split.validation]
    predictions = np.asarray(predictor.predict(features), dtype=np.float64)
    if predictions.shape != (len(split.validation), len(TARGET_COLUMNS)):
        raise AlertArtifactError(f"{model_id} validation prediction shape is invalid")
    truth = frame.iloc[split.validation].loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float64)
    table = pd.DataFrame(
        {
            "model_id": model_id,
            "scenario": split.name,
            "role": "validation",
            "row_index": split.validation,
            "v": frame.iloc[split.validation]["v"].to_numpy(),
        }
    )
    for index, target in enumerate(TARGET_COLUMNS):
        table[f"{target}_true"] = truth[:, index]
        table[f"{target}_predicted"] = predictions[:, index]
        table[f"{target}_error"] = predictions[:, index] - truth[:, index]
    return table


def _load_m2_bundle(
    frame: pd.DataFrame,
    splits: SplitMapping,
    artifact_directory: Path,
) -> ModelPredictionBundle:
    payload = _load_json(artifact_directory / "evaluation.json")
    split_hashes = compute_split_hashes(splits)
    _verify_manifest_common(
        payload,
        split_hashes=split_hashes,
        expected_versions=_current_m2_versions(),
        label="M2",
    )
    model_path = artifact_directory / "baseline_model.joblib"
    prediction_path = artifact_directory / "predictions" / "test_predictions.csv.gz"
    model_hash = _verify_hash(model_path, payload.get("model_sha256"), label="M2 model")
    prediction_hash = _verify_hash(
        prediction_path,
        payload.get("predictions_sha256"),
        label="M2 predictions",
    )
    predictor = cast(Predictor, joblib.load(model_path))
    test_predictions = pd.read_csv(prediction_path)
    validate_prediction_table(frame, test_predictions, splits)
    validation_predictions = _build_prediction_table(
        frame,
        splits["state_group"],
        predictor,
        model_id=MODEL_IDS[0],
    )
    test_predictions.insert(0, "model_id", MODEL_IDS[0])
    return ModelPredictionBundle(
        model_id=MODEL_IDS[0],
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
        model_sha256=model_hash,
        predictions_sha256=prediction_hash,
    )


def _load_m3_bundle(
    frame: pd.DataFrame,
    splits: SplitMapping,
    artifact_directory: Path,
) -> ModelPredictionBundle:
    payload = _load_json(artifact_directory / "evaluation.json")
    split_hashes = compute_split_hashes(splits)
    expected_versions = {**_current_m2_versions(), "torch": str(torch.__version__)}
    _verify_manifest_common(
        payload,
        split_hashes=split_hashes,
        expected_versions=expected_versions,
        label="M3",
    )
    checkpoint_path = artifact_directory / "checkpoints" / "state_group_seed_42.pt"
    prediction_path = artifact_directory / "predictions" / "test_predictions.csv.gz"
    checkpoint_hash = _verify_hash(
        checkpoint_path,
        payload.get("primary_checkpoint_sha256"),
        label="M3 checkpoint",
    )
    prediction_hash = _verify_hash(
        prediction_path,
        payload.get("predictions_sha256"),
        label="M3 predictions",
    )
    checkpoint = load_torch_checkpoint(checkpoint_path)
    if checkpoint.torch_version != str(torch.__version__):
        raise AlertArtifactError("M3 checkpoint torch version differs")
    expected_metadata = {
        "scenario": "state_group",
        "role": "selection_validation",
        "seed": 42,
        "train_split_hash": split_hashes["state_group"]["train"],
        "validation_split_hash": split_hashes["state_group"]["validation"],
    }
    if checkpoint.metadata != expected_metadata:
        raise AlertArtifactError("M3 checkpoint metadata differs from state-group selection")
    test_predictions = pd.read_csv(prediction_path)
    validate_prediction_table(frame, test_predictions, splits)
    validation_predictions = _build_prediction_table(
        frame,
        splits["state_group"],
        checkpoint.regressor,
        model_id=MODEL_IDS[1],
    )
    if "model_id" in test_predictions.columns:
        test_predictions = test_predictions.drop(columns="model_id")
    if "candidate_id" in test_predictions.columns:
        test_predictions = test_predictions.drop(columns="candidate_id")
    test_predictions.insert(0, "model_id", MODEL_IDS[1])
    return ModelPredictionBundle(
        model_id=MODEL_IDS[1],
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
        model_sha256=checkpoint_hash,
        predictions_sha256=prediction_hash,
    )


def load_verified_prediction_bundles(
    frame: pd.DataFrame,
    splits: SplitMapping,
    *,
    m2_artifact_directory: Path,
    m3_artifact_directory: Path,
) -> tuple[ModelPredictionBundle, ...]:
    """Load trusted M2/M3 validation and stored test predictions without retraining."""
    return (
        _load_m2_bundle(frame, splits, m2_artifact_directory),
        _load_m3_bundle(frame, splits, m3_artifact_directory),
    )


def _channel_scores(table: pd.DataFrame, channel: AlertChannel) -> tuple[np.ndarray, np.ndarray]:
    true_severity = coefficients_to_severity(_prediction_values(table, "true"))
    predicted_severity = coefficients_to_severity(_prediction_values(table, "predicted"))
    return channel_severity(true_severity, channel), channel_severity(
        predicted_severity,
        channel,
    )


def _metric_row(
    metrics: AlertClassificationMetrics,
    *,
    model_id: str,
    scenario: str,
    role: str,
    channel: AlertChannel,
    policy: str,
    target_fpr: float | None = None,
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "scenario": scenario,
        "role": role,
        "channel": channel,
        "policy": policy,
        "target_fpr": target_fpr,
        **metrics.to_dict(),
    }


def build_threshold_sensitivity(
    bundles: tuple[ModelPredictionBundle, ...],
    thresholds: tuple[float, ...] = SENSITIVITY_THRESHOLDS,
) -> pd.DataFrame:
    """Evaluate every pre-fixed threshold on state validation and all test scenarios."""
    rows = []
    for bundle in bundles:
        tables = (bundle.validation_predictions, bundle.test_predictions)
        for table in tables:
            for (scenario, role), group in table.groupby(["scenario", "role"], sort=True):
                for threshold in thresholds:
                    for channel_value in ALERT_CHANNELS:
                        channel = parse_alert_channel(channel_value)
                        reference, predicted = _channel_scores(group, channel)
                        metrics = compute_alert_metrics(
                            reference,
                            predicted,
                            reference_threshold=threshold,
                            decision_threshold=threshold,
                        )
                        rows.append(
                            _metric_row(
                                metrics,
                                model_id=bundle.model_id,
                                scenario=scenario,
                                role=role,
                                channel=channel,
                                policy="nominal_threshold",
                            )
                        )
    return pd.DataFrame(rows)


def build_fixed_fpr_cutoffs(
    bundles: tuple[ModelPredictionBundle, ...],
    target_fprs: tuple[float, ...] = FIXED_FPR_TARGETS,
) -> pd.DataFrame:
    """Calibrate model/channel cutoffs on state-group validation only."""
    rows = []
    for bundle in bundles:
        validation = bundle.validation_predictions
        for channel_value in ALERT_CHANNELS:
            channel = parse_alert_channel(channel_value)
            reference, predicted = _channel_scores(validation, channel)
            for target_fpr in target_fprs:
                cutoff = select_fixed_fpr_cutoff(
                    reference,
                    predicted,
                    target_fpr=target_fpr,
                )
                rows.append(
                    {
                        "model_id": bundle.model_id,
                        "scenario": "state_group",
                        "role": "validation",
                        "channel": channel,
                        "reference_threshold": ALERT_SEVERITY_THRESHOLD,
                        **cutoff.to_dict(),
                    }
                )
    return pd.DataFrame(rows)


def _cutoff_lookup(
    cutoffs: pd.DataFrame,
    *,
    model_id: str,
    channel: AlertChannel,
    target_fpr: float,
) -> FixedFprCutoff:
    matches = cutoffs.loc[
        (cutoffs["model_id"] == model_id)
        & (cutoffs["channel"] == channel)
        & np.isclose(cutoffs["target_fpr"], target_fpr, rtol=0.0, atol=1e-12)
    ]
    if len(matches) != 1:
        raise AlertArtifactError("No unique validation fixed-FPR cutoff is available")
    row = matches.iloc[0]
    return FixedFprCutoff(
        target_fpr=float(row["target_fpr"]),
        cutoff=float(row["cutoff"]),
        validation_recall=float(row["validation_recall"]),
        validation_fpr=float(row["validation_fpr"]),
        validation_precision=float(row["validation_precision"]),
    )


def build_fixed_fpr_test_metrics(
    bundles: tuple[ModelPredictionBundle, ...],
    cutoffs: pd.DataFrame,
    target_fprs: tuple[float, ...] = FIXED_FPR_TARGETS,
) -> pd.DataFrame:
    """Apply validation-frozen cutoffs unchanged to every test scenario."""
    rows = []
    for bundle in bundles:
        for scenario, group in bundle.test_predictions.groupby("scenario", sort=True):
            for channel_value in ALERT_CHANNELS:
                channel = parse_alert_channel(channel_value)
                reference, predicted = _channel_scores(group, channel)
                for target_fpr in target_fprs:
                    cutoff = _cutoff_lookup(
                        cutoffs,
                        model_id=bundle.model_id,
                        channel=channel,
                        target_fpr=target_fpr,
                    )
                    metrics = compute_alert_metrics(
                        reference,
                        predicted,
                        reference_threshold=ALERT_SEVERITY_THRESHOLD,
                        decision_threshold=cutoff.cutoff,
                    )
                    rows.append(
                        _metric_row(
                            metrics,
                            model_id=bundle.model_id,
                            scenario=scenario,
                            role="test",
                            channel=channel,
                            policy="validation_fixed_fpr",
                            target_fpr=target_fpr,
                        )
                    )
    return pd.DataFrame(rows)


def build_primary_policy_predictions(
    bundles: tuple[ModelPredictionBundle, ...],
) -> pd.DataFrame:
    """Build ignored row-level state outputs for the frozen primary policy threshold."""
    tables = []
    for bundle in bundles:
        for source in (bundle.validation_predictions, bundle.test_predictions):
            table = source.copy()
            true_severity = coefficients_to_severity(_prediction_values(table, "true"))
            predicted_severity = coefficients_to_severity(_prediction_values(table, "predicted"))
            for channel_value in ALERT_CHANNELS:
                channel = parse_alert_channel(channel_value)
                true_scores = channel_severity(true_severity, channel)
                predicted_scores = channel_severity(predicted_severity, channel)
                table[f"{channel}_true_severity"] = true_scores
                table[f"{channel}_predicted_severity"] = predicted_scores
                table[f"{channel}_reference_alert"] = true_scores >= ALERT_SEVERITY_THRESHOLD
                table[f"{channel}_predicted_alert"] = predicted_scores >= ALERT_SEVERITY_THRESHOLD
            tables.append(table)
    return pd.concat(tables, ignore_index=True)
