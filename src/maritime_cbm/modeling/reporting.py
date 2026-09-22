"""Tabular and graphical reports for M2 baseline experiments."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from maritime_cbm.data.schema import TARGET_COLUMNS
from maritime_cbm.modeling.evaluation import EvaluationRole, ScenarioEvaluation, SelectionOutcome
from maritime_cbm.modeling.metrics import state_group_selection_key


def build_metric_table(evaluations: tuple[ScenarioEvaluation, ...]) -> pd.DataFrame:
    """Flatten scenario evaluations into a stable long-form metric table."""
    rows = []
    for evaluation in evaluations:
        parameters = json.dumps(
            evaluation.candidate.parameter_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for role in evaluation.roles:
            for metric in role.metrics:
                rows.append(
                    {
                        "candidate_id": evaluation.candidate.candidate_id,
                        "family": evaluation.candidate.family,
                        "preprocessing": evaluation.candidate.preprocessing,
                        "parameters": parameters,
                        "selectable": evaluation.candidate.selectable,
                        "scenario": evaluation.scenario,
                        "role": role.role,
                        **metric.to_dict(),
                        "fit_seconds": evaluation.fit_seconds,
                        "prediction_seconds": role.prediction_seconds,
                    }
                )
    return pd.DataFrame(rows)


def build_selection_summary(outcome: SelectionOutcome) -> pd.DataFrame:
    """Summarize the state-group-only ranking used to select the final candidate."""
    rows = []
    state_group_evaluations = tuple(
        result for result in outcome.evaluations if result.scenario == "state_group"
    )
    for evaluation in state_group_evaluations:
        key = state_group_selection_key(
            evaluation.role("validation").metrics,
            complexity_rank=evaluation.candidate.complexity_rank,
            candidate_id=evaluation.candidate.candidate_id,
        )
        rows.append(
            {
                "candidate_id": evaluation.candidate.candidate_id,
                "selectable": evaluation.candidate.selectable,
                "mean_nrmse": key[0],
                "worst_target_nrmse": key[1],
                "complexity_rank": evaluation.candidate.complexity_rank,
                "selected": evaluation.candidate == outcome.selected_candidate,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["selectable", "mean_nrmse", "worst_target_nrmse", "complexity_rank", "candidate_id"],
        ascending=[False, True, True, True, True],
        ignore_index=True,
    )


def build_prediction_table(
    frame: pd.DataFrame,
    evaluations: tuple[ScenarioEvaluation, ...],
    *,
    role_name: EvaluationRole = "test",
) -> pd.DataFrame:
    """Build unaggregated predictions for ignored local artifact storage."""
    tables = []
    for evaluation in evaluations:
        role = evaluation.role(role_name)
        truth = frame.iloc[role.row_indices].loc[:, TARGET_COLUMNS].to_numpy()
        table = pd.DataFrame(
            {
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


def build_error_by_speed(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate test errors by scenario, speed and target."""
    rows = []
    for (scenario, speed), group in predictions.groupby(["scenario", "v"], sort=True):
        for target in TARGET_COLUMNS:
            errors = group[f"{target}_error"].to_numpy()
            rows.append(
                {
                    "scenario": scenario,
                    "v": speed,
                    "target": target,
                    "sample_count": len(group),
                    "mae": float(np.mean(np.abs(errors))),
                    "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                    "bias": float(np.mean(errors)),
                }
            )
    return pd.DataFrame(rows)


def build_error_by_target_state(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate test errors by the true value of each degradation coefficient."""
    rows = []
    for scenario, scenario_frame in predictions.groupby("scenario", sort=True):
        for target in TARGET_COLUMNS:
            for target_value, group in scenario_frame.groupby(f"{target}_true", sort=True):
                errors = group[f"{target}_error"].to_numpy()
                rows.append(
                    {
                        "scenario": scenario,
                        "target": target,
                        "target_value": target_value,
                        "sample_count": len(group),
                        "mae": float(np.mean(np.abs(errors))),
                        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                        "bias": float(np.mean(errors)),
                    }
                )
    return pd.DataFrame(rows)


def write_model_figures(predictions: pd.DataFrame, output_directory: Path) -> tuple[Path, ...]:
    """Write three compact test-error figures with a non-interactive backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)

    state_group = predictions.loc[predictions["scenario"] == "state_group"]
    prediction_path = output_directory / "state_group_predictions.png"
    figure, axes = plt.subplots(ncols=2, figsize=(10, 4))
    for axis, target in zip(axes, TARGET_COLUMNS, strict=True):
        actual = state_group[f"{target}_true"]
        predicted = state_group[f"{target}_predicted"]
        limits = (min(actual.min(), predicted.min()), max(actual.max(), predicted.max()))
        axis.scatter(actual, predicted, s=8, alpha=0.35)
        axis.plot(limits, limits, color="black", linestyle="--", linewidth=1)
        axis.set(title=target, xlabel="True", ylabel="Predicted")
    figure.suptitle("State-group test predictions")
    figure.tight_layout()
    figure.savefig(prediction_path, dpi=160)
    plt.close(figure)

    holdout_path = output_directory / "holdout_extrapolation.png"
    figure, axes = plt.subplots(ncols=2, figsize=(10, 4))
    holdout_targets = (("compressor_holdout", "kMc"), ("turbine_holdout", "kMt"))
    for axis, (scenario, target) in zip(axes, holdout_targets, strict=True):
        subset = predictions.loc[predictions["scenario"] == scenario]
        actual = subset[f"{target}_true"]
        predicted = subset[f"{target}_predicted"]
        limits = (min(actual.min(), predicted.min()), max(actual.max(), predicted.max()))
        axis.scatter(actual, predicted, s=8, alpha=0.35)
        axis.plot(limits, limits, color="black", linestyle="--", linewidth=1)
        axis.set(title=f"{scenario}: {target}", xlabel="True", ylabel="Predicted")
    figure.suptitle("Severe-degradation holdout tests")
    figure.tight_layout()
    figure.savefig(holdout_path, dpi=160)
    plt.close(figure)

    speed_path = output_directory / "state_group_error_by_speed.png"
    speed_errors = build_error_by_speed(state_group)
    figure, axes = plt.subplots(ncols=2, figsize=(10, 4), sharex=True)
    for axis, target in zip(axes, TARGET_COLUMNS, strict=True):
        subset = speed_errors.loc[speed_errors["target"] == target]
        axis.plot(subset["v"], subset["mae"], marker="o")
        axis.set(title=target, xlabel="Ship speed (knots)", ylabel="MAE")
    figure.suptitle("State-group test error by speed")
    figure.tight_layout()
    figure.savefig(speed_path, dpi=160)
    plt.close(figure)

    return prediction_path, holdout_path, speed_path
