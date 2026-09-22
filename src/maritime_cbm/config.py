"""Project configuration loaded from explicit environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RANDOM_SEED = 42
DEFAULT_RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "uci_cbm"
DEFAULT_EDA_REPORT_DIR = PROJECT_ROOT / "reports" / "eda"
DEFAULT_MODEL_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "modeling"
DEFAULT_MODEL_REPORT_DIR = PROJECT_ROOT / "reports" / "modeling"


class ConfigurationError(ValueError):
    """Raised when a project setting is invalid."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings shared by data and modeling modules."""

    raw_data_dir: Path
    model_artifact_dir: Path
    model_report_dir: Path
    random_seed: int


def _resolve_path(environment_name: str, default: Path) -> Path:
    value = os.getenv(environment_name)
    path = Path(value).expanduser() if value else default
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_settings() -> Settings:
    """Return settings using environment overrides when provided."""
    raw_data_dir = _resolve_path("MARITIME_CBM_RAW_DATA_DIR", DEFAULT_RAW_DATA_DIR)
    model_artifact_dir = _resolve_path(
        "MARITIME_CBM_MODEL_ARTIFACT_DIR", DEFAULT_MODEL_ARTIFACT_DIR
    )
    model_report_dir = _resolve_path("MARITIME_CBM_MODEL_REPORT_DIR", DEFAULT_MODEL_REPORT_DIR)

    seed_value = os.getenv("MARITIME_CBM_RANDOM_SEED", str(DEFAULT_RANDOM_SEED))
    try:
        random_seed = int(seed_value)
    except ValueError as error:
        raise ConfigurationError("MARITIME_CBM_RANDOM_SEED must be an integer") from error
    if random_seed < 0:
        raise ConfigurationError("MARITIME_CBM_RANDOM_SEED must be non-negative")

    return Settings(
        raw_data_dir=raw_data_dir,
        model_artifact_dir=model_artifact_dir,
        model_report_dir=model_report_dir,
        random_seed=random_seed,
    )
