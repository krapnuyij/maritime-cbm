"""Leakage-safe preprocessing, deterministic training and M3 checkpoints."""

import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from maritime_cbm.data.schema import MODEL_FEATURE_COLUMNS, TARGET_COLUMNS
from maritime_cbm.modeling.metrics import (
    TargetRegressionMetrics,
    compute_regression_metrics,
    state_group_selection_key,
)
from maritime_cbm.modeling.preprocessing import (
    SPEED_CENTERED_COLUMNS,
    ModelingInputError,
    SpeedConditionedCenterer,
)
from maritime_cbm.modeling.torch_models import (
    TorchCandidateSpec,
    TorchPreprocessing,
    build_torch_model,
    count_trainable_parameters,
    torch_candidate_from_dict,
)

LOGGER = logging.getLogger(__name__)
CHECKPOINT_SCHEMA_VERSION = 1
type DevicePreference = Literal["cpu", "mps", "auto"]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Frozen M3 optimization and early-stopping settings."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 500
    min_epochs: int = 50
    patience: int = 40
    min_delta: float = 1e-5

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Learning rate must be positive and weight decay non-negative")
        if self.batch_size <= 0 or self.max_epochs <= 0:
            raise ValueError("Batch size and maximum epochs must be positive")
        if self.min_epochs <= 0 or self.min_epochs > self.max_epochs:
            raise ValueError("Minimum epochs must be within the maximum epoch range")
        if self.patience <= 0 or self.min_delta < 0:
            raise ValueError("Patience must be positive and minimum delta non-negative")

    def to_dict(self) -> dict[str, float | int]:
        """Return a serialization-safe training configuration."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DeviceResolution:
    """Requested and resolved torch device with an optional fallback reason."""

    requested: DevicePreference
    resolved: Literal["cpu", "mps"]
    fallback_reason: str | None = None


def resolve_torch_device(preference: DevicePreference) -> DeviceResolution:
    """Resolve optional MPS execution while guaranteeing a CPU fallback."""
    if preference == "cpu":
        return DeviceResolution(requested=preference, resolved="cpu")
    if preference not in ("mps", "auto"):
        raise ValueError(f"Unsupported torch device preference: {preference}")
    if torch.backends.mps.is_available():
        return DeviceResolution(requested=preference, resolved="mps")
    return DeviceResolution(
        requested=preference,
        resolved="cpu",
        fallback_reason="MPS is not available; using CPU",
    )


def configure_torch_determinism(seed: int) -> None:
    """Configure the approved deterministic CPU-oriented training behavior."""
    if seed < 0:
        raise ValueError("Torch seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _canonical_feature_frame(values: object) -> pd.DataFrame:
    if not isinstance(values, pd.DataFrame):
        raise ModelingInputError("Torch model features must be a pandas DataFrame")
    missing_columns = tuple(
        column for column in MODEL_FEATURE_COLUMNS if column not in values.columns
    )
    if missing_columns:
        raise ModelingInputError(f"Required model columns are missing: {missing_columns}")
    frame = values.loc[:, MODEL_FEATURE_COLUMNS].copy()
    try:
        array = frame.to_numpy(dtype=np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise ModelingInputError("Torch model features must be numeric") from error
    if not np.isfinite(array).all():
        raise ModelingInputError("Torch model features must contain only finite values")
    return frame


def _target_array(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(TARGET_COLUMNS) or len(array) == 0:
        raise ModelingInputError(
            f"Torch targets must have shape (n_samples, {len(TARGET_COLUMNS)})"
        )
    if not np.isfinite(array).all():
        raise ModelingInputError("Torch targets must contain only finite values")
    return array


@dataclass(frozen=True, slots=True)
class FittedTorchPreprocessor:
    """Train-only speed centering and feature/target standardization state."""

    preprocessing: TorchPreprocessing
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    speed_values: np.ndarray
    speed_centers: np.ndarray

    @classmethod
    def fit(
        cls,
        features: pd.DataFrame,
        targets: object,
        *,
        preprocessing: TorchPreprocessing,
    ) -> "FittedTorchPreprocessor":
        """Fit all preprocessing statistics on one training role only."""
        frame = _canonical_feature_frame(features)
        target_values = _target_array(targets)
        if len(frame) != len(target_values):
            raise ModelingInputError("Torch features and targets must have equal row counts")

        if preprocessing == "speed_centered":
            centerer = SpeedConditionedCenterer().fit(frame)
            transformed = centerer.transform(frame).to_numpy(dtype=np.float64, copy=False)
            speed_values = centerer.known_speeds_.copy()
            speed_centers = centerer.speed_means_.loc[:, list(SPEED_CENTERED_COLUMNS)].to_numpy(
                dtype=np.float64, copy=True
            )
        elif preprocessing == "raw":
            transformed = frame.to_numpy(dtype=np.float64, copy=False)
            speed_values = np.empty(0, dtype=np.float64)
            speed_centers = np.empty((0, len(SPEED_CENTERED_COLUMNS)), dtype=np.float64)
        else:  # pragma: no cover - closed literal type validated by candidate construction.
            raise ValueError(f"Unsupported torch preprocessing: {preprocessing}")

        feature_scaler = StandardScaler().fit(transformed)
        target_scaler = StandardScaler().fit(target_values)
        return cls(
            preprocessing=preprocessing,
            feature_mean=np.asarray(feature_scaler.mean_, dtype=np.float64),
            feature_scale=np.asarray(feature_scaler.scale_, dtype=np.float64),
            target_mean=np.asarray(target_scaler.mean_, dtype=np.float64),
            target_scale=np.asarray(target_scaler.scale_, dtype=np.float64),
            speed_values=speed_values,
            speed_centers=speed_centers,
        )

    def _center_features(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.to_numpy(dtype=np.float64, copy=True)
        if self.preprocessing == "raw":
            return values

        speed_index = MODEL_FEATURE_COLUMNS.index("v")
        centered_indices = tuple(
            MODEL_FEATURE_COLUMNS.index(column) for column in SPEED_CENTERED_COLUMNS
        )
        center_index = {float(speed): index for index, speed in enumerate(self.speed_values)}
        row_center_indices = []
        unknown_speeds = []
        for speed in values[:, speed_index]:
            index = center_index.get(float(speed))
            if index is None:
                unknown_speeds.append(float(speed))
            else:
                row_center_indices.append(index)
        if unknown_speeds:
            raise ModelingInputError(
                f"No training statistics are available for speeds: {sorted(set(unknown_speeds))}"
            )
        values[:, centered_indices] -= self.speed_centers[np.asarray(row_center_indices)]
        return values

    def transform_features(self, features: pd.DataFrame) -> np.ndarray:
        """Apply train-only feature preprocessing and return float32 values."""
        frame = _canonical_feature_frame(features)
        centered = self._center_features(frame)
        return np.asarray((centered - self.feature_mean) / self.feature_scale, dtype=np.float32)

    def transform_targets(self, targets: object) -> np.ndarray:
        """Standardize both degradation coefficients using train-only statistics."""
        values = _target_array(targets)
        return np.asarray((values - self.target_mean) / self.target_scale, dtype=np.float32)

    def inverse_transform_targets(self, targets: object) -> np.ndarray:
        """Convert standardized target predictions back to original units."""
        values = _target_array(targets)
        return np.asarray(values * self.target_scale + self.target_mean, dtype=np.float64)

    def to_payload(self) -> dict[str, object]:
        """Return primitive preprocessing state suitable for weights-only loading."""
        return {
            "preprocessing": self.preprocessing,
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_scale": self.target_scale.tolist(),
            "speed_values": self.speed_values.tolist(),
            "speed_centers": self.speed_centers.tolist(),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "FittedTorchPreprocessor":
        """Validate and reconstruct preprocessing state from a checkpoint payload."""
        if not isinstance(payload, dict):
            raise ValueError("Torch preprocessor payload must be a dictionary")
        preprocessing_value = payload.get("preprocessing")
        if preprocessing_value not in ("raw", "speed_centered"):
            raise ValueError("Torch preprocessor payload has an invalid preprocessing kind")
        preprocessing = cast(TorchPreprocessing, preprocessing_value)

        try:
            feature_mean = np.asarray(payload["feature_mean"], dtype=np.float64)
            feature_scale = np.asarray(payload["feature_scale"], dtype=np.float64)
            target_mean = np.asarray(payload["target_mean"], dtype=np.float64)
            target_scale = np.asarray(payload["target_scale"], dtype=np.float64)
            speed_values = np.asarray(payload["speed_values"], dtype=np.float64)
            speed_centers = np.asarray(payload["speed_centers"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Torch preprocessor payload is incomplete") from error

        expected_feature_shape = (len(MODEL_FEATURE_COLUMNS),)
        expected_target_shape = (len(TARGET_COLUMNS),)
        if (
            feature_mean.shape != expected_feature_shape
            or feature_scale.shape != expected_feature_shape
        ):
            raise ValueError("Torch preprocessor feature scaler shape is invalid")
        if (
            target_mean.shape != expected_target_shape
            or target_scale.shape != expected_target_shape
        ):
            raise ValueError("Torch preprocessor target scaler shape is invalid")
        if not len(speed_values) and speed_centers.size == 0:
            speed_centers = speed_centers.reshape(0, len(SPEED_CENTERED_COLUMNS))
        if speed_centers.shape != (len(speed_values), len(SPEED_CENTERED_COLUMNS)):
            raise ValueError("Torch preprocessor speed-centering shape is invalid")
        numeric_arrays = (
            feature_mean,
            feature_scale,
            target_mean,
            target_scale,
            speed_values,
            speed_centers,
        )
        if not all(np.isfinite(array).all() for array in numeric_arrays):
            raise ValueError("Torch preprocessor payload must contain finite values")
        if np.any(feature_scale <= 0) or np.any(target_scale <= 0):
            raise ValueError("Torch preprocessor scales must be positive")

        return cls(
            preprocessing=preprocessing,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            target_mean=target_mean,
            target_scale=target_scale,
            speed_values=speed_values,
            speed_centers=speed_centers,
        )


@dataclass(frozen=True, slots=True)
class EpochRecord:
    """One epoch of optimization and validation history."""

    epoch: int
    train_loss: float
    validation_mean_nrmse: float


class EarlyStopping:
    """Track meaningful validation improvements and restore the best state."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.best_score = float("inf")
        self.best_epoch = 0
        self.steps_without_improvement = 0
        self._best_state: dict[str, torch.Tensor] | None = None

    def update(self, epoch: int, score: float, model: nn.Module) -> bool:
        """Capture an improved state and return whether training should stop."""
        if epoch <= 0 or not np.isfinite(score):
            raise ValueError("Early-stopping epoch and score must be valid")
        if score < self.best_score - self.config.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.steps_without_improvement = 0
            self._best_state = {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            }
        else:
            self.steps_without_improvement += 1
        return (
            epoch >= self.config.min_epochs
            and self.steps_without_improvement >= self.config.patience
        )

    def restore(self, model: nn.Module) -> None:
        """Restore the captured parameters into a model on its current device."""
        if self._best_state is None:
            raise RuntimeError("Early stopping has no checkpoint to restore")
        model.load_state_dict(self._best_state)


@dataclass(slots=True)
class FittedTorchRegressor:
    """CPU inference bundle containing a model and fitted preprocessing state."""

    candidate: TorchCandidateSpec
    model: nn.Module
    preprocessor: FittedTorchPreprocessor

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return unclipped degradation-coefficient predictions in original units."""
        transformed = self.preprocessor.transform_features(features)
        self.model.eval()
        with torch.no_grad():
            scaled_predictions = self.model(torch.from_numpy(transformed)).cpu().numpy()
        return self.preprocessor.inverse_transform_targets(scaled_predictions)


@dataclass(slots=True)
class TorchTrainingResult:
    """A fitted inference bundle plus reproducibility and learning-curve metadata."""

    regressor: FittedTorchRegressor
    config: TrainingConfig
    seed: int
    requested_device: DevicePreference
    resolved_device: Literal["cpu", "mps"]
    fallback_reason: str | None
    best_epoch: int
    best_validation_metrics: tuple[TargetRegressionMetrics, ...]
    history: tuple[EpochRecord, ...]
    training_seconds: float
    trainable_parameter_count: int


def _run_training(
    train_features: pd.DataFrame,
    train_targets: np.ndarray,
    validation_features: pd.DataFrame,
    validation_targets: np.ndarray,
    *,
    candidate: TorchCandidateSpec,
    preprocessor: FittedTorchPreprocessor,
    seed: int,
    config: TrainingConfig,
    device_name: Literal["cpu", "mps"],
) -> tuple[
    FittedTorchRegressor,
    int,
    tuple[TargetRegressionMetrics, ...],
    tuple[EpochRecord, ...],
    float,
    int,
]:
    configure_torch_determinism(seed)
    device = torch.device(device_name)
    model = build_torch_model(candidate).to(device)
    parameter_count = count_trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_function = nn.MSELoss()

    train_x = torch.from_numpy(preprocessor.transform_features(train_features))
    train_y = torch.from_numpy(preprocessor.transform_targets(train_targets))
    validation_x = torch.from_numpy(preprocessor.transform_features(validation_features)).to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    early_stopping = EarlyStopping(config)
    history = []
    best_metrics: tuple[TargetRegressionMetrics, ...] | None = None
    started = perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_features)
            loss = loss_function(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(batch_features)
            sample_count += len(batch_features)

        model.eval()
        with torch.no_grad():
            scaled_validation = model(validation_x).detach().cpu().numpy()
        validation_predictions = preprocessor.inverse_transform_targets(scaled_validation)
        validation_metrics = compute_regression_metrics(
            validation_targets,
            validation_predictions,
        )
        validation_score = state_group_selection_key(
            validation_metrics,
            complexity_rank=0,
            candidate_id=candidate.candidate_id,
        )[0]
        should_stop = early_stopping.update(epoch, validation_score, model)
        if early_stopping.best_epoch == epoch:
            best_metrics = validation_metrics
        history.append(
            EpochRecord(
                epoch=epoch,
                train_loss=loss_sum / sample_count,
                validation_mean_nrmse=validation_score,
            )
        )
        if should_stop:
            break

    training_seconds = perf_counter() - started
    early_stopping.restore(model)
    model = model.cpu()
    if best_metrics is None:  # pragma: no cover - first finite epoch always improves infinity.
        raise RuntimeError("Training completed without a best validation checkpoint")
    regressor = FittedTorchRegressor(
        candidate=candidate,
        model=model,
        preprocessor=preprocessor,
    )
    return (
        regressor,
        early_stopping.best_epoch,
        best_metrics,
        tuple(history),
        training_seconds,
        parameter_count,
    )


def fit_torch_regressor(
    train_features: pd.DataFrame,
    train_targets: object,
    validation_features: pd.DataFrame,
    validation_targets: object,
    *,
    candidate: TorchCandidateSpec,
    seed: int,
    config: TrainingConfig | None = None,
    device: DevicePreference = "cpu",
) -> TorchTrainingResult:
    """Fit one deterministic candidate without accessing any test role."""
    resolved_config = TrainingConfig() if config is None else config
    train_target_values = _target_array(train_targets)
    validation_target_values = _target_array(validation_targets)
    preprocessor = FittedTorchPreprocessor.fit(
        train_features,
        train_target_values,
        preprocessing=candidate.preprocessing,
    )
    resolution = resolve_torch_device(device)
    fallback_reason = resolution.fallback_reason
    resolved_device = resolution.resolved

    try:
        training_values = _run_training(
            train_features,
            train_target_values,
            validation_features,
            validation_target_values,
            candidate=candidate,
            preprocessor=preprocessor,
            seed=seed,
            config=resolved_config,
            device_name=resolved_device,
        )
    except RuntimeError as error:
        if resolved_device != "mps":
            raise
        fallback_reason = f"MPS training failed; using CPU: {error}"
        LOGGER.warning(fallback_reason)
        resolved_device = "cpu"
        training_values = _run_training(
            train_features,
            train_target_values,
            validation_features,
            validation_target_values,
            candidate=candidate,
            preprocessor=preprocessor,
            seed=seed,
            config=resolved_config,
            device_name=resolved_device,
        )

    regressor, best_epoch, best_metrics, history, seconds, parameter_count = training_values
    return TorchTrainingResult(
        regressor=regressor,
        config=resolved_config,
        seed=seed,
        requested_device=device,
        resolved_device=resolved_device,
        fallback_reason=fallback_reason,
        best_epoch=best_epoch,
        best_validation_metrics=best_metrics,
        history=history,
        training_seconds=seconds,
        trainable_parameter_count=parameter_count,
    )


@dataclass(slots=True)
class LoadedTorchCheckpoint:
    """Weights-only loaded inference bundle and immutable training metadata."""

    regressor: FittedTorchRegressor
    training_config: TrainingConfig
    seed: int
    resolved_device: str
    best_epoch: int
    trainable_parameter_count: int
    metadata: dict[str, object]


def save_torch_checkpoint(
    result: TorchTrainingResult,
    path: Path,
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    """Save tensors and primitive metadata for weights-only loading."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "candidate": result.regressor.candidate.to_dict(),
        "training_config": result.config.to_dict(),
        "seed": result.seed,
        "requested_device": result.requested_device,
        "resolved_device": result.resolved_device,
        "fallback_reason": result.fallback_reason,
        "best_epoch": result.best_epoch,
        "trainable_parameter_count": result.trainable_parameter_count,
        "preprocessor": result.regressor.preprocessor.to_payload(),
        "model_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in result.regressor.model.state_dict().items()
        },
        "torch_version": str(torch.__version__),
        "metadata": {} if metadata is None else dict(metadata),
    }
    torch.save(payload, path)


def load_torch_checkpoint(path: Path) -> LoadedTorchCheckpoint:
    """Load a trusted M3 checkpoint with PyTorch weights-only restrictions."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Torch checkpoint must contain a dictionary")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Torch checkpoint schema version is not supported")

    candidate = torch_candidate_from_dict(payload.get("candidate"))
    preprocessor = FittedTorchPreprocessor.from_payload(payload.get("preprocessor"))
    config_payload = payload.get("training_config")
    if not isinstance(config_payload, dict):
        raise ValueError("Torch checkpoint has no valid training configuration")
    training_config = TrainingConfig(**config_payload)

    model = build_torch_model(candidate)
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, dict) or not all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in model_state.items()
    ):
        raise ValueError("Torch checkpoint has no valid model state")
    model.load_state_dict(model_state)
    model.eval()

    parameter_count = payload.get("trainable_parameter_count")
    if not isinstance(parameter_count, int) or parameter_count != count_trainable_parameters(model):
        raise ValueError("Torch checkpoint parameter count does not match the candidate")
    seed = payload.get("seed")
    best_epoch = payload.get("best_epoch")
    resolved_device = payload.get("resolved_device")
    metadata = payload.get("metadata")
    if not isinstance(seed, int) or not isinstance(best_epoch, int):
        raise ValueError("Torch checkpoint training metadata is invalid")
    if not isinstance(resolved_device, str) or not isinstance(metadata, dict):
        raise ValueError("Torch checkpoint environment metadata is invalid")

    return LoadedTorchCheckpoint(
        regressor=FittedTorchRegressor(
            candidate=candidate,
            model=model,
            preprocessor=preprocessor,
        ),
        training_config=training_config,
        seed=seed,
        resolved_device=resolved_device,
        best_epoch=best_epoch,
        trainable_parameter_count=parameter_count,
        metadata=metadata,
    )
