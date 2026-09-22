"""Run separated M3 neural selection and frozen-checkpoint final evaluation."""

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch

from maritime_cbm.config import get_settings
from maritime_cbm.data.loader import compute_sha256, load_raw_dataset
from maritime_cbm.data.splitting import build_dataset_splits, compute_split_hashes
from maritime_cbm.data.validation import DatasetValidationReport, validate_release
from maritime_cbm.logging_config import configure_logging
from maritime_cbm.modeling.reporting import (
    build_error_by_speed,
    build_error_by_target_state,
)
from maritime_cbm.modeling.torch_evaluation import (
    SELECTION_SCENARIOS,
    TorchSelectionOutcome,
    build_selection_seeds,
    run_torch_final_evaluation,
    run_torch_model_selection,
)
from maritime_cbm.modeling.torch_models import (
    TorchCandidateSpec,
    build_torch_candidate_specs,
    torch_candidate_from_dict,
)
from maritime_cbm.modeling.torch_reporting import (
    build_baseline_comparison,
    build_holdout_diagnostics,
    build_selected_training_history,
    build_torch_final_metrics,
    build_torch_prediction_table,
    build_torch_selection_metrics,
    build_torch_selection_summary,
    write_torch_figures,
)
from maritime_cbm.modeling.torch_training import (
    DevicePreference,
    LoadedTorchCheckpoint,
    TrainingConfig,
    load_torch_checkpoint,
    save_torch_checkpoint,
)

SELECTION_SCHEMA_VERSION = 1
SELECTION_PROTOCOL = "state_group_validation_three_seed_mean_nrmse"


class TorchSelectionManifestError(ValueError):
    """Raised when a frozen M3 selection cannot be reproduced or trusted."""


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "torch": str(torch.__version__),
    }


def _runtime_environment() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }


def _release_hashes(validation_report: DatasetValidationReport) -> dict[str, str]:
    return {item.relative_path: item.sha256 for item in validation_report.files}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise TorchSelectionManifestError(f"Cannot load M3 selection manifest: {path}") from error
    if not isinstance(payload, dict):
        raise TorchSelectionManifestError("M3 selection manifest must contain a JSON object")
    return payload


def _checkpoint_filename(scenario: str, seed: int) -> str:
    return f"checkpoints/{scenario}_seed_{seed}.pt"


def _save_selected_checkpoints(
    outcome: TorchSelectionOutcome,
    artifact_directory: Path,
    *,
    seed: int,
) -> tuple[dict[str, str], dict[str, str]]:
    checkpoint_files = {}
    checkpoint_hashes = {}
    for scenario in SELECTION_SCENARIOS:
        evaluation = outcome.selected_evaluation(scenario, seed)
        relative_path = _checkpoint_filename(scenario, seed)
        checkpoint_path = artifact_directory / relative_path
        save_torch_checkpoint(
            evaluation.training_result,
            checkpoint_path,
            metadata={
                "scenario": scenario,
                "role": "selection_validation",
                "seed": seed,
                "train_split_hash": outcome.split_hashes[scenario]["train"],
                "validation_split_hash": outcome.split_hashes[scenario]["validation"],
            },
        )
        checkpoint_files[scenario] = relative_path
        checkpoint_hashes[scenario] = compute_sha256(checkpoint_path)
    return checkpoint_files, checkpoint_hashes


def _selection_payload(
    outcome: TorchSelectionOutcome,
    *,
    base_seed: int,
    release_hashes: dict[str, str],
    config: TrainingConfig,
    device: DevicePreference,
    checkpoint_files: dict[str, str],
    checkpoint_hashes: dict[str, str],
) -> dict[str, object]:
    key = outcome.selection_key
    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "selection_protocol": SELECTION_PROTOCOL,
        "created_at": datetime.now(UTC).isoformat(),
        "base_seed": base_seed,
        "selection_seeds": list(outcome.selection_seeds),
        "release_hashes": release_hashes,
        "split_hashes": outcome.split_hashes,
        "runtime_versions": _runtime_versions(),
        "runtime_environment": _runtime_environment(),
        "device_preference": device,
        "training_config": config.to_dict(),
        "candidate_grid": [candidate.to_dict() for candidate in build_torch_candidate_specs()],
        "candidate_count": len(build_torch_candidate_specs()),
        "selected_candidate": outcome.selected_candidate.to_dict(),
        "selection_key": {
            "mean_nrmse": key[0],
            "worst_target_nrmse": key[1],
            "trainable_parameter_count": key[2],
            "architecture_rank": key[3],
            "candidate_id": key[4],
        },
        "checkpoint_files": checkpoint_files,
        "checkpoint_hashes": checkpoint_hashes,
    }


def _validate_selection_manifest(
    payload: dict[str, object],
    *,
    base_seed: int,
    release_hashes: dict[str, str],
    split_hashes: dict[str, dict[str, str]],
    config: TrainingConfig,
    device: DevicePreference,
) -> TorchCandidateSpec:
    expected_pairs: dict[str, object] = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "selection_protocol": SELECTION_PROTOCOL,
        "base_seed": base_seed,
        "selection_seeds": list(build_selection_seeds(base_seed)),
        "release_hashes": release_hashes,
        "split_hashes": split_hashes,
        "runtime_versions": _runtime_versions(),
        "runtime_environment": _runtime_environment(),
        "device_preference": device,
        "training_config": config.to_dict(),
        "candidate_grid": [candidate.to_dict() for candidate in build_torch_candidate_specs()],
        "candidate_count": len(build_torch_candidate_specs()),
    }
    for key, expected in expected_pairs.items():
        if payload.get(key) != expected:
            raise TorchSelectionManifestError(
                f"M3 selection manifest {key!r} does not match the evaluation environment"
            )
    try:
        return torch_candidate_from_dict(payload.get("selected_candidate"))
    except ValueError as error:
        raise TorchSelectionManifestError(str(error)) from error


def _load_selection_checkpoints(
    payload: dict[str, object],
    artifact_directory: Path,
    *,
    candidate: TorchCandidateSpec,
    seed: int,
    config: TrainingConfig,
) -> dict[str, LoadedTorchCheckpoint]:
    checkpoint_files = payload.get("checkpoint_files")
    checkpoint_hashes = payload.get("checkpoint_hashes")
    if not isinstance(checkpoint_files, dict) or not isinstance(checkpoint_hashes, dict):
        raise TorchSelectionManifestError("M3 selection manifest has no checkpoint mapping")
    if set(checkpoint_files) != set(SELECTION_SCENARIOS) or set(checkpoint_hashes) != set(
        SELECTION_SCENARIOS
    ):
        raise TorchSelectionManifestError("M3 checkpoint mapping has unexpected scenarios")

    loaded = {}
    artifact_root = artifact_directory.resolve()
    for scenario in SELECTION_SCENARIOS:
        relative_path = checkpoint_files[scenario]
        expected_hash = checkpoint_hashes[scenario]
        if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
            raise TorchSelectionManifestError("M3 checkpoint mapping contains invalid values")
        checkpoint_path = (artifact_root / relative_path).resolve()
        if not checkpoint_path.is_relative_to(artifact_root):
            raise TorchSelectionManifestError("M3 checkpoint path escapes the artifact directory")
        if compute_sha256(checkpoint_path) != expected_hash:
            raise TorchSelectionManifestError(f"M3 checkpoint hash differs for {scenario}")
        checkpoint = load_torch_checkpoint(checkpoint_path)
        expected_metadata = {
            "scenario": scenario,
            "role": "selection_validation",
            "seed": seed,
            "train_split_hash": payload["split_hashes"][scenario]["train"],
            "validation_split_hash": payload["split_hashes"][scenario]["validation"],
        }
        if checkpoint.regressor.candidate != candidate:
            raise TorchSelectionManifestError(f"M3 checkpoint candidate differs for {scenario}")
        if checkpoint.seed != seed or checkpoint.training_config != config:
            raise TorchSelectionManifestError(
                f"M3 checkpoint training setup differs for {scenario}"
            )
        if checkpoint.torch_version != _runtime_versions()["torch"]:
            raise TorchSelectionManifestError(f"M3 checkpoint torch version differs for {scenario}")
        if checkpoint.metadata != expected_metadata:
            raise TorchSelectionManifestError(f"M3 checkpoint metadata differs for {scenario}")
        loaded[scenario] = checkpoint
    return loaded


def run_selection(
    data_directory: Path,
    artifact_directory: Path,
    report_directory: Path,
    *,
    seed: int,
    device: DevicePreference,
) -> dict[str, object]:
    """Execute the frozen M3 grid without evaluating any test role."""
    validation_report = validate_release(data_directory)
    frame = load_raw_dataset(data_directory / "data.txt")
    config = TrainingConfig()
    outcome = run_torch_model_selection(
        frame,
        base_seed=seed,
        config=config,
        device=device,
    )

    artifact_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_files, checkpoint_hashes = _save_selected_checkpoints(
        outcome,
        artifact_directory,
        seed=seed,
    )
    payload = _selection_payload(
        outcome,
        base_seed=seed,
        release_hashes=_release_hashes(validation_report),
        config=config,
        device=device,
        checkpoint_files=checkpoint_files,
        checkpoint_hashes=checkpoint_hashes,
    )

    selection_path = artifact_directory / "selection.json"
    metrics_path = report_directory / "candidate_validation_metrics.csv"
    summary_path = report_directory / "selection_summary.csv"
    history_path = report_directory / "selected_training_history.csv"
    _write_json(selection_path, payload)
    build_torch_selection_metrics(outcome).to_csv(metrics_path, index=False)
    build_torch_selection_summary(outcome).to_csv(summary_path, index=False)
    build_selected_training_history(outcome, seed=seed).to_csv(history_path, index=False)
    return {
        "selection_manifest": str(selection_path),
        "validation_metrics": str(metrics_path),
        "selection_summary": str(summary_path),
        "training_history": str(history_path),
        "selected_candidate": outcome.selected_candidate.candidate_id,
        "selection_key": payload["selection_key"],
        "checkpoints": checkpoint_files,
    }


def run_evaluation(
    data_directory: Path,
    artifact_directory: Path,
    report_directory: Path,
    *,
    selection_path: Path,
    baseline_metrics_path: Path,
    seed: int,
    device: DevicePreference,
    include_figures: bool,
) -> dict[str, object]:
    """Evaluate frozen seed-42 checkpoints and compare them with committed M2 metrics."""
    validation_report = validate_release(data_directory)
    frame = load_raw_dataset(data_directory / "data.txt")
    payload = _load_json(selection_path)
    config = TrainingConfig()
    split_hashes = compute_split_hashes(build_dataset_splits(frame, seed=seed))
    candidate = _validate_selection_manifest(
        payload,
        base_seed=seed,
        release_hashes=_release_hashes(validation_report),
        split_hashes=split_hashes,
        config=config,
        device=device,
    )
    checkpoints = _load_selection_checkpoints(
        payload,
        artifact_directory,
        candidate=candidate,
        seed=seed,
        config=config,
    )
    outcome = run_torch_final_evaluation(
        frame,
        candidate,
        checkpoints,
        seed=seed,
        config=config,
        device=device,
    )

    artifact_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)
    predictions_directory = artifact_directory / "predictions"
    predictions_directory.mkdir(parents=True, exist_ok=True)

    metrics_path = report_directory / "final_metrics.csv"
    predictions_path = predictions_directory / "test_predictions.csv.gz"
    comparison_path = report_directory / "baseline_comparison.csv"
    holdout_path = report_directory / "holdout_diagnostics.csv"
    error_by_speed_path = report_directory / "test_error_by_speed.csv"
    error_by_state_path = report_directory / "test_error_by_target_state.csv"
    training_history_path = report_directory / "selected_training_history.csv"

    metrics = build_torch_final_metrics(outcome)
    predictions = build_torch_prediction_table(frame, outcome)
    baseline_metrics = pd.read_csv(baseline_metrics_path)
    comparison = build_baseline_comparison(baseline_metrics, metrics)
    holdout_diagnostics = build_holdout_diagnostics(predictions)
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False, compression="gzip")
    comparison.to_csv(comparison_path, index=False)
    holdout_diagnostics.to_csv(holdout_path, index=False)
    build_error_by_speed(predictions).to_csv(error_by_speed_path, index=False)
    build_error_by_target_state(predictions).to_csv(error_by_state_path, index=False)

    figure_paths = ()
    if include_figures:
        training_history = pd.read_csv(training_history_path)
        figure_paths = write_torch_figures(
            training_history,
            comparison,
            predictions,
            report_directory,
        )

    primary_checkpoint = artifact_directory / _checkpoint_filename("state_group", seed)
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "selected_candidate": candidate.candidate_id,
        "selection_manifest": str(selection_path),
        "primary_checkpoint": str(primary_checkpoint),
        "primary_checkpoint_sha256": compute_sha256(primary_checkpoint),
        "baseline_metrics": str(baseline_metrics_path),
        "baseline_metrics_sha256": compute_sha256(baseline_metrics_path),
        "predictions": str(predictions_path),
        "predictions_sha256": compute_sha256(predictions_path),
        "metrics": str(metrics_path),
        "comparison": str(comparison_path),
        "holdout_diagnostics": str(holdout_path),
        "error_by_speed": str(error_by_speed_path),
        "error_by_target_state": str(error_by_state_path),
        "figures": [str(path) for path in figure_paths],
        "split_hashes": outcome.split_hashes,
        "runtime_versions": _runtime_versions(),
        "runtime_environment": _runtime_environment(),
    }
    evaluation_path = artifact_directory / "evaluation.json"
    _write_json(evaluation_path, result)
    result["evaluation_manifest"] = str(evaluation_path)
    return result


def _build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("select", "evaluate"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument(
            "--data-directory",
            type=Path,
            default=settings.raw_data_dir,
        )
        command_parser.add_argument(
            "--artifact-directory",
            type=Path,
            default=settings.m3_model_artifact_dir,
        )
        command_parser.add_argument(
            "--report-directory",
            type=Path,
            default=settings.m3_model_report_dir,
        )
        command_parser.add_argument(
            "--device",
            choices=("cpu", "mps", "auto"),
            default="cpu",
        )

    evaluation_parser = subparsers.choices["evaluate"]
    evaluation_parser.add_argument("--selection-file", type=Path)
    evaluation_parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=settings.model_report_dir / "final_metrics.csv",
    )
    evaluation_parser.add_argument("--skip-figures", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one explicitly separated M3 experiment stage."""
    configure_logging()
    settings = get_settings()
    arguments = _build_parser().parse_args(argv)
    device = arguments.device
    if arguments.command == "select":
        result = run_selection(
            arguments.data_directory,
            arguments.artifact_directory,
            arguments.report_directory,
            seed=settings.random_seed,
            device=device,
        )
    else:
        selection_path = arguments.selection_file or arguments.artifact_directory / "selection.json"
        result = run_evaluation(
            arguments.data_directory,
            arguments.artifact_directory,
            arguments.report_directory,
            selection_path=selection_path,
            baseline_metrics_path=arguments.baseline_metrics,
            seed=settings.random_seed,
            device=device,
            include_figures=not arguments.skip_figures,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
