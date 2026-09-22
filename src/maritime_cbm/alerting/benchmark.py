"""Run verified downstream M4 alert-policy evaluation without model retraining."""

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import pandas as pd

from maritime_cbm.alerting.evaluation import (
    build_fixed_fpr_cutoffs,
    build_fixed_fpr_test_metrics,
    build_primary_policy_predictions,
    build_threshold_sensitivity,
    load_verified_prediction_bundles,
)
from maritime_cbm.alerting.reporting import (
    build_primary_alert_metrics,
    build_primary_model_comparison,
    write_alert_figures,
)
from maritime_cbm.config import get_settings
from maritime_cbm.data.loader import compute_sha256, load_raw_dataset
from maritime_cbm.data.splitting import build_dataset_splits, compute_split_hashes
from maritime_cbm.data.validation import validate_release
from maritime_cbm.logging_config import configure_logging


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_deterministic_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write gzip bytes without filename or timestamp metadata."""
    buffer = BytesIO()
    frame.to_csv(
        buffer,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    path.write_bytes(buffer.getvalue())


def run_alert_evaluation(
    data_directory: Path,
    m2_artifact_directory: Path,
    m3_artifact_directory: Path,
    artifact_directory: Path,
    report_directory: Path,
    *,
    seed: int,
    include_figures: bool,
) -> dict[str, object]:
    """Verify upstream artifacts and write fixed-policy aggregate reports."""
    validate_release(data_directory)
    frame = load_raw_dataset(data_directory / "data.txt")
    splits = build_dataset_splits(frame, seed=seed)
    bundles = load_verified_prediction_bundles(
        frame,
        splits,
        m2_artifact_directory=m2_artifact_directory,
        m3_artifact_directory=m3_artifact_directory,
    )

    sensitivity = build_threshold_sensitivity(bundles)
    primary = build_primary_alert_metrics(sensitivity)
    comparison = build_primary_model_comparison(primary)
    cutoffs = build_fixed_fpr_cutoffs(bundles)
    fixed_fpr_metrics = build_fixed_fpr_test_metrics(bundles, cutoffs)
    row_predictions = build_primary_policy_predictions(bundles)

    artifact_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)
    sensitivity_path = report_directory / "threshold_sensitivity.csv"
    primary_path = report_directory / "primary_alert_metrics.csv"
    comparison_path = report_directory / "primary_model_comparison.csv"
    cutoffs_path = report_directory / "fixed_fpr_cutoffs.csv"
    fixed_fpr_path = report_directory / "fixed_fpr_test_metrics.csv"
    predictions_path = artifact_directory / "policy_predictions.csv.gz"

    sensitivity.to_csv(sensitivity_path, index=False)
    primary.to_csv(primary_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    cutoffs.to_csv(cutoffs_path, index=False)
    fixed_fpr_metrics.to_csv(fixed_fpr_path, index=False)
    _write_deterministic_gzip_csv(row_predictions, predictions_path)
    figure_paths = (
        write_alert_figures(sensitivity, primary, report_directory) if include_figures else ()
    )

    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "policy": {
            "severity_thresholds": [0.5, 0.6, 0.7, 0.8, 0.9],
            "primary_alert_threshold": 0.8,
            "fixed_fpr_targets": [0.01, 0.05],
        },
        "source_artifacts": {
            bundle.model_id: {
                "model_sha256": bundle.model_sha256,
                "predictions_sha256": bundle.predictions_sha256,
            }
            for bundle in bundles
        },
        "split_hashes": compute_split_hashes(splits),
        "row_predictions": str(predictions_path),
        "row_predictions_sha256": compute_sha256(predictions_path),
        "reports": [
            str(path)
            for path in (
                sensitivity_path,
                primary_path,
                comparison_path,
                cutoffs_path,
                fixed_fpr_path,
            )
        ],
        "figures": [str(path) for path in figure_paths],
    }
    manifest_path = artifact_directory / "evaluation.json"
    _write_json(manifest_path, result)
    result["evaluation_manifest"] = str(manifest_path)
    return result


def _build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=settings.raw_data_dir)
    parser.add_argument(
        "--m2-artifact-directory",
        type=Path,
        default=settings.model_artifact_dir,
    )
    parser.add_argument(
        "--m3-artifact-directory",
        type=Path,
        default=settings.m3_model_artifact_dir,
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=settings.alert_artifact_dir,
    )
    parser.add_argument(
        "--report-directory",
        type=Path,
        default=settings.alert_report_dir,
    )
    parser.add_argument("--skip-figures", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixed M4 downstream evaluation once."""
    configure_logging()
    settings = get_settings()
    arguments = _build_parser().parse_args(argv)
    result = run_alert_evaluation(
        arguments.data_directory,
        arguments.m2_artifact_directory,
        arguments.m3_artifact_directory,
        arguments.artifact_directory,
        arguments.report_directory,
        seed=settings.random_seed,
        include_figures=not arguments.skip_figures,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
