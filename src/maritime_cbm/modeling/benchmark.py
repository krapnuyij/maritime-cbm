"""Run validation-only baseline selection and frozen-candidate final evaluation."""

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

from maritime_cbm.config import get_settings
from maritime_cbm.data.loader import compute_sha256, load_raw_dataset
from maritime_cbm.data.splitting import build_dataset_splits, compute_split_hashes
from maritime_cbm.data.validation import DatasetValidationReport, validate_release
from maritime_cbm.logging_config import configure_logging
from maritime_cbm.modeling.evaluation import (
    SelectionOutcome,
    run_final_evaluation,
    run_model_selection,
)
from maritime_cbm.modeling.models import CandidateSpec, get_candidate_spec
from maritime_cbm.modeling.reporting import (
    build_error_by_speed,
    build_error_by_target_state,
    build_metric_table,
    build_prediction_table,
    build_selection_summary,
    write_model_figures,
)

SELECTION_SCHEMA_VERSION = 1
SELECTION_PROTOCOL = "state_group_validation_mean_nrmse"


class SelectionManifestError(ValueError):
    """Raised when a frozen selection cannot be reproduced for final evaluation."""


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
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
        raise SelectionManifestError(f"Cannot load selection manifest: {path}") from error
    if not isinstance(payload, dict):
        raise SelectionManifestError("Selection manifest must contain a JSON object")
    return payload


def _selection_payload(
    outcome: SelectionOutcome,
    *,
    seed: int,
    release_hashes: dict[str, str],
) -> dict[str, object]:
    selected_candidate = outcome.selected_candidate
    selection_key = outcome.selection_key
    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "selection_protocol": SELECTION_PROTOCOL,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "release_hashes": release_hashes,
        "split_hashes": outcome.split_hashes,
        "runtime_versions": _runtime_versions(),
        "candidate_count": len({item.candidate for item in outcome.evaluations}),
        "selected_candidate": selected_candidate.to_dict(),
        "selection_key": {
            "mean_nrmse": selection_key[0],
            "worst_target_nrmse": selection_key[1],
            "complexity_rank": selection_key[2],
            "candidate_id": selection_key[3],
        },
    }


def _validate_selection_manifest(
    payload: dict[str, object],
    *,
    seed: int,
    release_hashes: dict[str, str],
    split_hashes: dict[str, dict[str, str]],
) -> CandidateSpec:
    expected_pairs = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "selection_protocol": SELECTION_PROTOCOL,
        "seed": seed,
        "release_hashes": release_hashes,
        "split_hashes": split_hashes,
        "runtime_versions": _runtime_versions(),
    }
    for key, expected in expected_pairs.items():
        if payload.get(key) != expected:
            raise SelectionManifestError(
                f"Selection manifest {key!r} does not match the evaluation environment"
            )

    stored_candidate = payload.get("selected_candidate")
    if not isinstance(stored_candidate, dict):
        raise SelectionManifestError("Selection manifest has no valid selected_candidate")
    candidate_id = stored_candidate.get("candidate_id")
    if not isinstance(candidate_id, str):
        raise SelectionManifestError("Selected candidate has no valid candidate_id")
    try:
        candidate = get_candidate_spec(candidate_id)
    except KeyError as error:
        raise SelectionManifestError(str(error)) from error
    if stored_candidate != candidate.to_dict():
        raise SelectionManifestError("Selected candidate differs from the approved candidate grid")
    return candidate


def run_selection(
    data_directory: Path,
    artifact_directory: Path,
    report_directory: Path,
    *,
    seed: int,
) -> dict[str, object]:
    """Execute candidate selection without evaluating any test role."""
    validation_report = validate_release(data_directory)
    frame = load_raw_dataset(data_directory / "data.txt")
    outcome = run_model_selection(frame, seed=seed)
    payload = _selection_payload(
        outcome,
        seed=seed,
        release_hashes=_release_hashes(validation_report),
    )

    artifact_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)
    selection_path = artifact_directory / "selection.json"
    metrics_path = report_directory / "candidate_validation_metrics.csv"
    summary_path = report_directory / "model_selection_summary.csv"
    _write_json(selection_path, payload)
    build_metric_table(outcome.evaluations).to_csv(metrics_path, index=False)
    build_selection_summary(outcome).to_csv(summary_path, index=False)
    return {
        "selection_manifest": str(selection_path),
        "validation_metrics": str(metrics_path),
        "selection_summary": str(summary_path),
        "selected_candidate": outcome.selected_candidate.candidate_id,
        "selection_key": payload["selection_key"],
    }


def run_evaluation(
    data_directory: Path,
    artifact_directory: Path,
    report_directory: Path,
    *,
    selection_path: Path,
    seed: int,
    include_figures: bool,
) -> dict[str, object]:
    """Evaluate a frozen selected candidate once on all approved test scenarios."""
    validation_report = validate_release(data_directory)
    frame = load_raw_dataset(data_directory / "data.txt")
    manifest = _load_json(selection_path)

    current_split_hashes = compute_split_hashes(build_dataset_splits(frame, seed=seed))
    candidate = _validate_selection_manifest(
        manifest,
        seed=seed,
        release_hashes=_release_hashes(validation_report),
        split_hashes=current_split_hashes,
    )
    outcome = run_final_evaluation(frame, candidate, seed=seed)

    artifact_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)
    predictions_directory = artifact_directory / "predictions"
    predictions_directory.mkdir(parents=True, exist_ok=True)

    metrics_path = report_directory / "final_metrics.csv"
    predictions_path = predictions_directory / "test_predictions.csv.gz"
    error_by_speed_path = report_directory / "test_error_by_speed.csv"
    error_by_state_path = report_directory / "test_error_by_target_state.csv"
    model_path = artifact_directory / "baseline_model.joblib"

    build_metric_table(outcome.evaluations).to_csv(metrics_path, index=False)
    predictions = build_prediction_table(frame, outcome.evaluations)
    predictions.to_csv(predictions_path, index=False, compression="gzip")
    build_error_by_speed(predictions).to_csv(error_by_speed_path, index=False)
    build_error_by_target_state(predictions).to_csv(error_by_state_path, index=False)
    joblib.dump(outcome.primary_estimator, model_path)
    figure_paths = write_model_figures(predictions, report_directory) if include_figures else ()

    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "selected_candidate": candidate.candidate_id,
        "selection_manifest": str(selection_path),
        "model": str(model_path),
        "model_sha256": compute_sha256(model_path),
        "predictions": str(predictions_path),
        "predictions_sha256": compute_sha256(predictions_path),
        "metrics": str(metrics_path),
        "error_by_speed": str(error_by_speed_path),
        "error_by_target_state": str(error_by_state_path),
        "figures": [str(path) for path in figure_paths],
        "split_hashes": outcome.split_hashes,
        "runtime_versions": _runtime_versions(),
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
            default=settings.model_artifact_dir,
        )
        command_parser.add_argument(
            "--report-directory",
            type=Path,
            default=settings.model_report_dir,
        )

    evaluation_parser = subparsers.choices["evaluate"]
    evaluation_parser.add_argument("--selection-file", type=Path)
    evaluation_parser.add_argument("--skip-figures", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one explicitly separated M2 experiment stage."""
    configure_logging()
    settings = get_settings()
    arguments = _build_parser().parse_args(argv)
    if arguments.command == "select":
        result = run_selection(
            arguments.data_directory,
            arguments.artifact_directory,
            arguments.report_directory,
            seed=settings.random_seed,
        )
    else:
        selection_path = arguments.selection_file or arguments.artifact_directory / "selection.json"
        result = run_evaluation(
            arguments.data_directory,
            arguments.artifact_directory,
            arguments.report_directory,
            selection_path=selection_path,
            seed=settings.random_seed,
            include_figures=not arguments.skip_figures,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
