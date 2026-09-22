"""Tracked aggregate tables and figures for M4 alert-policy evaluation."""

from pathlib import Path

import numpy as np
import pandas as pd

from maritime_cbm.alerting.policy import ALERT_SEVERITY_THRESHOLD


def build_primary_alert_metrics(sensitivity: pd.DataFrame) -> pd.DataFrame:
    """Select primary-threshold test rows from the full sensitivity table."""
    rows = sensitivity.loc[
        (sensitivity["role"] == "test")
        & np.isclose(
            sensitivity["reference_threshold"],
            ALERT_SEVERITY_THRESHOLD,
            rtol=0.0,
            atol=1e-12,
        )
    ].copy()
    return rows.sort_values(["scenario", "channel", "model_id"], ignore_index=True)


def build_primary_model_comparison(primary_metrics: pd.DataFrame) -> pd.DataFrame:
    """Place M2 and M3 primary policy metrics side by side without hiding NA values."""
    metric_columns = (
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "f1",
        "fpr",
        "miss_rate",
        "pr_auc",
        "reference_prevalence",
        "metric_valid",
        "unavailable_reason",
    )
    keys = ["scenario", "channel"]
    m2 = primary_metrics.loc[
        primary_metrics["model_id"] == "m2_random_forest",
        keys + list(metric_columns),
    ]
    m3 = primary_metrics.loc[
        primary_metrics["model_id"] == "m3_linear_residual_mlp",
        keys + list(metric_columns),
    ]
    comparison = m2.merge(
        m3,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_m2", "_m3"),
    )
    if len(comparison) != 12:
        raise ValueError("Primary comparison requires four scenarios and three channels")
    for metric in ("recall", "f1", "fpr", "miss_rate", "pr_auc"):
        comparison[f"{metric}_delta_m3_minus_m2"] = (
            comparison[f"{metric}_m3"] - comparison[f"{metric}_m2"]
        )
    return comparison.sort_values(keys, ignore_index=True)


def write_alert_figures(
    sensitivity: pd.DataFrame,
    primary_metrics: pd.DataFrame,
    output_directory: Path,
) -> tuple[Path, ...]:
    """Write threshold sensitivity and severe-holdout recall figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)

    sensitivity_path = output_directory / "state_group_threshold_sensitivity.png"
    state_group = sensitivity.loc[
        (sensitivity["scenario"] == "state_group") & (sensitivity["role"] == "test")
    ]
    figure, axes = plt.subplots(ncols=2, figsize=(11, 4))
    for (model_id, channel), group in state_group.groupby(["model_id", "channel"], sort=True):
        label = f"{model_id} · {channel}"
        axes[0].plot(group["reference_threshold"], group["recall"], marker="o", label=label)
        axes[1].plot(group["reference_threshold"], group["fpr"], marker="o", label=label)
    axes[0].set(title="Recall", xlabel="Severity threshold", ylabel="Recall", ylim=(-0.02, 1.02))
    axes[1].set(title="False-positive rate", xlabel="Severity threshold", ylabel="FPR")
    axes[1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.suptitle("State-group test alert sensitivity")
    figure.tight_layout()
    figure.savefig(sensitivity_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    holdout_path = output_directory / "severe_holdout_recall.png"
    selectors = (
        ("compressor_holdout", "kMc"),
        ("compressor_holdout", "any"),
        ("turbine_holdout", "kMt"),
        ("turbine_holdout", "any"),
    )
    rows = []
    for scenario, channel in selectors:
        rows.append(
            primary_metrics.loc[
                (primary_metrics["scenario"] == scenario) & (primary_metrics["channel"] == channel)
            ]
        )
    holdout = pd.concat(rows, ignore_index=True)
    holdout["label"] = holdout["scenario"] + "\n" + holdout["channel"]
    labels = list(dict.fromkeys(holdout["label"]))
    positions = np.arange(len(labels), dtype=np.float64)
    width = 0.38
    figure, axis = plt.subplots(figsize=(10, 4.5))
    model_ids = ("m2_random_forest", "m3_linear_residual_mlp")
    for offset, model_id in zip((-width / 2, width / 2), model_ids, strict=True):
        subset = holdout.loc[holdout["model_id"] == model_id].set_index("label").loc[labels]
        axis.bar(positions + offset, subset["recall"], width, label=model_id)
    axis.set_xticks(positions, labels)
    axis.set(ylabel="Recall", title="Primary policy severe-holdout recall", ylim=(0, 1.05))
    axis.legend()
    figure.tight_layout()
    figure.savefig(holdout_path, dpi=160)
    plt.close(figure)

    return sensitivity_path, holdout_path
