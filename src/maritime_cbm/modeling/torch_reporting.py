"""Tracked tables and figures for M3 neural-model experiments."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from maritime_cbm.data.schema import TARGET_COLUMNS
from maritime_cbm.modeling.metrics import (
    TARGET_RANGES,
    aggregated_state_group_selection_key,
)
from maritime_cbm.modeling.torch_evaluation import (
    TorchFinalEvaluationOutcome,
    TorchSelectionOutcome,
)


def build_torch_selection_metrics(outcome: TorchSelectionOutcome) -> pd.DataFrame:
    """Flatten all candidate, scenario and seed validation metrics."""
    rows = []
    for evaluation in outcome.evaluations:
        result = evaluation.training_result
        parameters = json.dumps(
            evaluation.candidate.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for metric in evaluation.metrics:
            rows.append(
                {
                    "candidate_id": evaluation.candidate.candidate_id,
                    "architecture": evaluation.candidate.architecture,
                    "preprocessing": evaluation.candidate.preprocessing,
                    "parameters": parameters,
                    "scenario": evaluation.scenario,
                    "role": "validation",
                    "seed": evaluation.seed,
                    **metric.to_dict(),
                    "best_epoch": result.best_epoch,
                    "epochs_ran": len(result.history),
                    "training_seconds": result.training_seconds,
                    "trainable_parameter_count": result.trainable_parameter_count,
                    "requested_device": result.requested_device,
                    "resolved_device": result.resolved_device,
                    "fallback_reason": result.fallback_reason,
                }
            )
    return pd.DataFrame(rows)


def build_torch_selection_summary(outcome: TorchSelectionOutcome) -> pd.DataFrame:
    """Rank candidates after averaging each target NRMSE across seeds."""
    rows = []
    candidates = tuple(dict.fromkeys(result.candidate for result in outcome.evaluations))
    for candidate in candidates:
        state_group_runs = tuple(
            result
            for result in outcome.evaluations
            if result.candidate == candidate and result.scenario == "state_group"
        )
        parameter_count = state_group_runs[0].training_result.trainable_parameter_count
        key = aggregated_state_group_selection_key(
            tuple(result.metrics for result in state_group_runs),
            trainable_parameter_count=parameter_count,
            architecture_rank=candidate.architecture_rank,
            candidate_id=candidate.candidate_id,
        )
        nrmse_by_target = {
            target: np.asarray(
                [
                    next(metric.nrmse for metric in result.metrics if metric.target == target)
                    for result in state_group_runs
                ],
                dtype=np.float64,
            )
            for target in TARGET_COLUMNS
        }
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "architecture": candidate.architecture,
                "preprocessing": candidate.preprocessing,
                "hidden_sizes": json.dumps(candidate.hidden_sizes),
                "seed_count": len(state_group_runs),
                "kMc_nrmse_mean": float(np.mean(nrmse_by_target["kMc"])),
                "kMc_nrmse_std": float(np.std(nrmse_by_target["kMc"])),
                "kMt_nrmse_mean": float(np.mean(nrmse_by_target["kMt"])),
                "kMt_nrmse_std": float(np.std(nrmse_by_target["kMt"])),
                "mean_nrmse": key[0],
                "worst_target_nrmse": key[1],
                "trainable_parameter_count": key[2],
                "architecture_rank": key[3],
                "selected": candidate == outcome.selected_candidate,
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "mean_nrmse",
            "worst_target_nrmse",
            "trainable_parameter_count",
            "architecture_rank",
            "candidate_id",
        ],
        ignore_index=True,
    )


def build_selected_training_history(
    outcome: TorchSelectionOutcome,
    *,
    seed: int,
) -> pd.DataFrame:
    """Return learning curves for the selected seed-42 scenario checkpoints."""
    rows = []
    for scenario in ("state_group", "compressor_holdout", "turbine_holdout"):
        evaluation = outcome.selected_evaluation(scenario, seed)
        for record in evaluation.training_result.history:
            rows.append(
                {
                    "candidate_id": evaluation.candidate.candidate_id,
                    "scenario": scenario,
                    "seed": seed,
                    "epoch": record.epoch,
                    "train_loss": record.train_loss,
                    "validation_mean_nrmse": record.validation_mean_nrmse,
                    "best_epoch": evaluation.training_result.best_epoch,
                }
            )
    return pd.DataFrame(rows)


def build_torch_final_metrics(outcome: TorchFinalEvaluationOutcome) -> pd.DataFrame:
    """Flatten validation and one-time test metrics for the frozen neural candidate."""
    rows = []
    for evaluation in outcome.evaluations:
        for role in evaluation.roles:
            for metric in role.metrics:
                rows.append(
                    {
                        "candidate_id": evaluation.candidate.candidate_id,
                        "architecture": evaluation.candidate.architecture,
                        "preprocessing": evaluation.candidate.preprocessing,
                        "scenario": evaluation.scenario,
                        "role": role.role,
                        "seed": evaluation.seed,
                        **metric.to_dict(),
                        "training_source": evaluation.training_source,
                        "training_seconds": evaluation.training_seconds,
                        "best_epoch": evaluation.best_epoch,
                        "trainable_parameter_count": evaluation.trainable_parameter_count,
                        "prediction_seconds": role.prediction_seconds,
                    }
                )
    return pd.DataFrame(rows)


def build_torch_prediction_table(
    frame: pd.DataFrame,
    outcome: TorchFinalEvaluationOutcome,
) -> pd.DataFrame:
    """Build unaggregated M3 test predictions for ignored artifact storage."""
    tables = []
    for evaluation in outcome.evaluations:
        role = evaluation.role("test")
        truth = frame.iloc[role.row_indices].loc[:, TARGET_COLUMNS].to_numpy()
        table = pd.DataFrame(
            {
                "candidate_id": evaluation.candidate.candidate_id,
                "scenario": evaluation.scenario,
                "role": role.role,
                "row_index": role.row_indices,
                "v": frame.iloc[role.row_indices]["v"].to_numpy(),
            }
        )
        for index, target in enumerate(TARGET_COLUMNS):
            table[f"{target}_true"] = truth[:, index]
            table[f"{target}_predicted"] = role.predictions[:, index]
            table[f"{target}_error"] = role.predictions[:, index] - truth[:, index]
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def build_baseline_comparison(
    baseline_metrics: pd.DataFrame,
    torch_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Compare committed M2 and M3 test metrics without rerunning the baseline."""
    metric_columns = ("mae", "rmse", "r2", "bias", "absolute_error_p95", "nrmse")
    keys = ["scenario", "target"]
    baseline_test = baseline_metrics.loc[
        baseline_metrics["role"] == "test", keys + list(metric_columns)
    ]
    torch_test = torch_metrics.loc[torch_metrics["role"] == "test", keys + list(metric_columns)]
    if len(baseline_test) != 8 or len(torch_test) != 8:
        raise ValueError("Baseline and M3 comparison each require eight test metric rows")
    comparison = baseline_test.merge(
        torch_test,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_random_forest", "_pytorch"),
    )
    if len(comparison) != 8:
        raise ValueError("Baseline and M3 test metrics do not cover the same scenarios and targets")
    for metric in metric_columns:
        comparison[f"{metric}_delta"] = (
            comparison[f"{metric}_pytorch"] - comparison[f"{metric}_random_forest"]
        )
    comparison["nrmse_relative_change"] = (
        comparison["nrmse_delta"] / comparison["nrmse_random_forest"]
    )
    return comparison.sort_values(keys, ignore_index=True)


def build_holdout_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Quantify M3 extrapolation span, slope and bias for both severe-degradation tests."""
    rows = []
    for scenario, target in (
        ("compressor_holdout", "kMc"),
        ("turbine_holdout", "kMt"),
    ):
        subset = predictions.loc[predictions["scenario"] == scenario]
        truth = subset[f"{target}_true"].to_numpy(dtype=np.float64)
        predicted = subset[f"{target}_predicted"].to_numpy(dtype=np.float64)
        errors = predicted - truth
        true_span = float(np.ptp(truth))
        predicted_span = float(np.ptp(predicted))
        slope = float(np.polyfit(truth, predicted, deg=1)[0])
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        rows.append(
            {
                "scenario": scenario,
                "target": target,
                "sample_count": len(subset),
                "true_min": float(np.min(truth)),
                "true_max": float(np.max(truth)),
                "predicted_min": float(np.min(predicted)),
                "predicted_max": float(np.max(predicted)),
                "prediction_span_ratio": predicted_span / true_span,
                "prediction_slope": slope,
                "bias": float(np.mean(errors)),
                "rmse": rmse,
                "nrmse": rmse / TARGET_RANGES[target],
            }
        )
    return pd.DataFrame(rows)


def write_torch_figures(
    training_history: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
    predictions: pd.DataFrame,
    output_directory: Path,
) -> tuple[Path, ...]:
    """Write compact M3 learning-curve, baseline-comparison and holdout figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)

    learning_path = output_directory / "learning_curves.png"
    figure, axes = plt.subplots(ncols=2, figsize=(11, 4))
    for scenario, group in training_history.groupby("scenario", sort=True):
        axes[0].plot(group["epoch"], group["train_loss"], label=scenario)
        axes[1].plot(group["epoch"], group["validation_mean_nrmse"], label=scenario)
        best_epoch = int(group["best_epoch"].iloc[0])
        axes[1].axvline(best_epoch, linewidth=0.8, linestyle="--", alpha=0.4)
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Standardized MSE")
    axes[1].set(title="Validation score", xlabel="Epoch", ylabel="Mean NRMSE")
    axes[1].legend(fontsize=8)
    figure.suptitle("Selected M3 candidate learning curves (seed 42)")
    figure.tight_layout()
    figure.savefig(learning_path, dpi=160)
    plt.close(figure)

    comparison_path = output_directory / "baseline_comparison.png"
    ordered = baseline_comparison.copy()
    ordered["label"] = ordered["scenario"] + "\n" + ordered["target"]
    positions = np.arange(len(ordered))
    width = 0.38
    figure, axis = plt.subplots(figsize=(12, 4.5))
    axis.bar(
        positions - width / 2,
        ordered["nrmse_random_forest"],
        width,
        label="Random Forest",
    )
    axis.bar(
        positions + width / 2,
        ordered["nrmse_pytorch"],
        width,
        label="PyTorch",
    )
    axis.set_xticks(positions, ordered["label"], rotation=25, ha="right")
    axis.set(ylabel="NRMSE", title="M2 baseline and M3 test comparison")
    axis.legend()
    figure.tight_layout()
    figure.savefig(comparison_path, dpi=160)
    plt.close(figure)

    holdout_path = output_directory / "holdout_extrapolation.png"
    figure, axes = plt.subplots(ncols=2, figsize=(10, 4))
    for axis, (scenario, target) in zip(
        axes,
        (("compressor_holdout", "kMc"), ("turbine_holdout", "kMt")),
        strict=True,
    ):
        subset = predictions.loc[predictions["scenario"] == scenario]
        actual = subset[f"{target}_true"]
        predicted = subset[f"{target}_predicted"]
        limits = (min(actual.min(), predicted.min()), max(actual.max(), predicted.max()))
        axis.scatter(actual, predicted, s=8, alpha=0.35)
        axis.plot(limits, limits, color="black", linestyle="--", linewidth=1)
        axis.set(title=f"{scenario}: {target}", xlabel="True", ylabel="Predicted")
    figure.suptitle("M3 severe-degradation holdout tests")
    figure.tight_layout()
    figure.savefig(holdout_path, dpi=160)
    plt.close(figure)

    return learning_path, comparison_path, holdout_path
