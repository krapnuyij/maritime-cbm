"""Generate reproducible aggregate EDA tables and optional figures."""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from maritime_cbm.config import DEFAULT_EDA_REPORT_DIR, get_settings
from maritime_cbm.data.features import select_model_features
from maritime_cbm.data.loader import load_raw_dataset
from maritime_cbm.data.schema import (
    ALL_COLUMNS,
    EXPECTED_SPEED_VALUES,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMNS,
)
from maritime_cbm.data.splitting import (
    DatasetSplit,
    build_dataset_splits,
    compute_split_hashes,
    diagnose_group_neighbors,
)
from maritime_cbm.data.validation import validate_dataframe

SUMMARY_FILENAME = "summary_statistics.csv"
CORRELATION_FILENAME = "train_correlation_matrix.csv"
SPEED_CORRELATION_FILENAME = "train_speed_conditioned_correlations.csv"
SPEED_VARIANCE_FILENAME = "train_between_speed_variance_ratio.csv"
GRID_FIGURE_FILENAME = "target_grid_coverage.png"
GROUP_SPLIT_FIGURE_FILENAME = "state_group_split.png"
CORRELATION_FIGURE_FILENAME = "train_feature_target_correlations.png"
SENSOR_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    feature for feature in MODEL_FEATURE_COLUMNS if feature != "v"
)


@dataclass(frozen=True, slots=True)
class EdaReport:
    """Machine-readable summary of one EDA generation run."""

    output_directory: str
    table_files: tuple[str, ...]
    figure_files: tuple[str, ...]
    split_hashes: dict[str, dict[str, str]]
    group_neighbor_diagnostics: dict[str, float | int]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable report."""
        return asdict(self)


def build_summary_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize every raw input and target using the complete release."""
    summary = frame.loc[:, ALL_COLUMNS].describe().T
    summary.insert(1, "unique", frame.loc[:, ALL_COLUMNS].nunique(dropna=False))
    summary.index.name = "variable"
    return summary.reset_index()


def build_train_correlation_matrix(
    frame: pd.DataFrame,
    primary_split: DatasetSplit,
) -> pd.DataFrame:
    """Compute feature/target correlations using only primary-split training rows."""
    train_frame = frame.iloc[primary_split.train]
    model_features = select_model_features(train_frame)
    analysis_frame = pd.concat(
        [model_features, train_frame.loc[:, TARGET_COLUMNS].copy()],
        axis="columns",
    )
    correlation = analysis_frame.corr(method="pearson")
    correlation.index.name = "variable"
    return correlation.reset_index()


def build_speed_conditioned_correlations(
    frame: pd.DataFrame,
    primary_split: DatasetSplit,
) -> pd.DataFrame:
    """Compute train-only feature/target correlations separately at each ship speed."""
    train_frame = frame.iloc[primary_split.train]
    records: list[dict[str, float | str]] = []
    for speed, speed_frame in train_frame.groupby("v", sort=True):
        model_features = select_model_features(speed_frame)
        analysis_frame = pd.concat(
            [model_features, speed_frame.loc[:, TARGET_COLUMNS].copy()],
            axis="columns",
        )
        correlation = analysis_frame.corr(method="pearson")
        for feature in SENSOR_FEATURE_COLUMNS:
            for target in TARGET_COLUMNS:
                records.append(
                    {
                        "speed_knots": float(speed),
                        "feature": feature,
                        "target": target,
                        "correlation": float(correlation.loc[feature, target]),
                    }
                )
    return pd.DataFrame.from_records(
        records,
        columns=["speed_knots", "feature", "target", "correlation"],
    )


def build_between_speed_variance_ratios(
    frame: pd.DataFrame,
    primary_split: DatasetSplit,
) -> pd.DataFrame:
    """Measure the train feature variance explained by differences between speed means."""
    train_frame = frame.iloc[primary_split.train]
    records: list[dict[str, float | str]] = []
    for feature in SENSOR_FEATURE_COLUMNS:
        feature_values = train_frame[feature].to_numpy(dtype=np.float64, copy=False)
        overall_mean = float(np.mean(feature_values))
        total_variance = float(np.mean(np.square(feature_values - overall_mean)))
        speed_means = train_frame.groupby("v", sort=True)[feature].transform("mean").to_numpy()
        between_speed_variance = float(np.mean(np.square(speed_means - overall_mean)))
        ratio = between_speed_variance / total_variance if total_variance else 0.0
        records.append(
            {
                "feature": feature,
                "between_speed_variance_ratio": ratio,
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=["feature", "between_speed_variance_ratio"],
    )


def write_eda_tables(
    frame: pd.DataFrame,
    primary_split: DatasetSplit,
    output_directory: Path,
) -> tuple[Path, ...]:
    """Write small aggregate EDA tables without requiring plotting dependencies."""
    output_directory.mkdir(parents=True, exist_ok=True)
    summary_path = output_directory / SUMMARY_FILENAME
    correlation_path = output_directory / CORRELATION_FILENAME
    speed_correlation_path = output_directory / SPEED_CORRELATION_FILENAME
    speed_variance_path = output_directory / SPEED_VARIANCE_FILENAME
    build_summary_statistics(frame).to_csv(summary_path, index=False, float_format="%.10g")
    build_train_correlation_matrix(frame, primary_split).to_csv(
        correlation_path,
        index=False,
        float_format="%.10g",
    )
    build_speed_conditioned_correlations(frame, primary_split).to_csv(
        speed_correlation_path,
        index=False,
        float_format="%.10g",
    )
    build_between_speed_variance_ratios(frame, primary_split).to_csv(
        speed_variance_path,
        index=False,
        float_format="%.10g",
    )
    return summary_path, correlation_path, speed_correlation_path, speed_variance_path


def _import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "EDA figures require matplotlib; run `uv sync --group eda` first"
        ) from error
    return plt


def write_eda_figures(
    frame: pd.DataFrame,
    primary_split: DatasetSplit,
    output_directory: Path,
) -> tuple[Path, ...]:
    """Write the three approved EDA figures with English labels and the Agg backend."""
    plt = _import_pyplot()
    output_directory.mkdir(parents=True, exist_ok=True)

    grid_path = output_directory / GRID_FIGURE_FILENAME
    grid_counts = frame.groupby(["kMc", "kMt"], sort=True).size().unstack(fill_value=0)
    figure, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(
        grid_counts.to_numpy(),
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=len(EXPECTED_SPEED_VALUES),
        extent=(
            float(grid_counts.columns.min()) - 0.0005,
            float(grid_counts.columns.max()) + 0.0005,
            float(grid_counts.index.min()) - 0.0005,
            float(grid_counts.index.max()) + 0.0005,
        ),
    )
    axis.set_title("Complete degradation-state grid (9 speeds per pair)")
    axis.set_xlabel("kMt")
    axis.set_ylabel("kMc")
    figure.colorbar(image, ax=axis, label="Observation count")
    figure.tight_layout()
    figure.savefig(grid_path, dpi=160)
    plt.close(figure)

    group_split_path = output_directory / GROUP_SPLIT_FIGURE_FILENAME
    figure, axis = plt.subplots(figsize=(8, 5))
    colors = {"train": "#4C78A8", "validation": "#F2CF5B", "test": "#E45756"}
    for role, indices in primary_split.indices_by_role().items():
        states = frame.iloc[indices].loc[:, ["kMc", "kMt"]].drop_duplicates()
        axis.scatter(
            states["kMc"],
            states["kMt"],
            s=12,
            alpha=0.8,
            label=role,
            color=colors[role],
        )
    axis.set_title("Primary state-group split")
    axis.set_xlabel("kMc")
    axis.set_ylabel("kMt")
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    figure.tight_layout()
    figure.savefig(group_split_path, dpi=160)
    plt.close(figure)

    correlation_path = output_directory / CORRELATION_FIGURE_FILENAME
    correlation = build_train_correlation_matrix(frame, primary_split).set_index("variable")
    pooled = correlation.loc[list(SENSOR_FEATURE_COLUMNS), list(TARGET_COLUMNS)]
    speed_conditioned = build_speed_conditioned_correlations(frame, primary_split)
    speed_conditioned["absolute_correlation"] = speed_conditioned["correlation"].abs()
    median_absolute = speed_conditioned.pivot_table(
        index="feature",
        columns="target",
        values="absolute_correlation",
        aggfunc="median",
        sort=False,
    ).loc[list(SENSOR_FEATURE_COLUMNS), list(TARGET_COLUMNS)]

    figure, axes = plt.subplots(ncols=2, figsize=(10, 7), sharey=True)
    pooled_image = axes[0].imshow(
        pooled.to_numpy(),
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        aspect="auto",
    )
    axes[0].set_title("Pooled Pearson r")
    axes[0].set_xticks(np.arange(len(TARGET_COLUMNS)), labels=TARGET_COLUMNS)
    axes[0].set_yticks(np.arange(len(SENSOR_FEATURE_COLUMNS)), labels=SENSOR_FEATURE_COLUMNS)
    figure.colorbar(pooled_image, ax=axes[0], label="Correlation", fraction=0.046)

    conditioned_image = axes[1].imshow(
        median_absolute.to_numpy(),
        vmin=0,
        vmax=1,
        cmap="viridis",
        aspect="auto",
    )
    axes[1].set_title("Median |r| within each speed")
    axes[1].set_xticks(np.arange(len(TARGET_COLUMNS)), labels=TARGET_COLUMNS)
    figure.colorbar(conditioned_image, ax=axes[1], label="Absolute correlation", fraction=0.046)
    figure.suptitle("Train-only feature-target correlations")
    figure.tight_layout()
    figure.savefig(correlation_path, dpi=160)
    plt.close(figure)

    return grid_path, group_split_path, correlation_path


def generate_eda_report(
    data_directory: Path,
    output_directory: Path = DEFAULT_EDA_REPORT_DIR,
    *,
    include_figures: bool = True,
) -> EdaReport:
    """Validate the release, construct approved splits, and write aggregate EDA outputs."""
    frame = load_raw_dataset(data_directory / "data.txt")
    validate_dataframe(frame)
    splits = build_dataset_splits(frame)
    primary_split = splits["state_group"]
    table_paths = write_eda_tables(frame, primary_split, output_directory)
    figure_paths = (
        write_eda_figures(frame, primary_split, output_directory) if include_figures else ()
    )
    diagnostics = diagnose_group_neighbors(frame, primary_split)
    return EdaReport(
        output_directory=str(output_directory),
        table_files=tuple(path.name for path in table_paths),
        figure_files=tuple(path.name for path in figure_paths),
        split_hashes=compute_split_hashes(splits),
        group_neighbor_diagnostics={
            "test_group_count": diagnostics.test_group_count,
            "test_groups_with_eight_neighbor": diagnostics.test_groups_with_eight_neighbor,
            "eight_neighbor_percentage": diagnostics.eight_neighbor_percentage,
            "test_groups_with_four_neighbor": diagnostics.test_groups_with_four_neighbor,
            "four_neighbor_percentage": diagnostics.four_neighbor_percentage,
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_directory",
        nargs="?",
        type=Path,
        default=settings.raw_data_dir,
        help="Directory containing the official data.txt file",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_EDA_REPORT_DIR,
        help="Directory for aggregate CSV and PNG outputs",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Generate CSV tables without importing matplotlib",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate EDA outputs and print split reproducibility metadata as JSON."""
    arguments = _build_parser().parse_args(argv)
    report = generate_eda_report(
        arguments.data_directory,
        arguments.output_directory,
        include_figures=not arguments.skip_figures,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
