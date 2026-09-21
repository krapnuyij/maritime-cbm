"""Project configuration loaded from explicit environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RANDOM_SEED = 42
DEFAULT_RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "uci_cbm"
DEFAULT_EDA_REPORT_DIR = PROJECT_ROOT / "reports" / "eda"


class ConfigurationError(ValueError):
    """Raised when a project setting is invalid."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings shared by data and modeling modules."""

    raw_data_dir: Path
    random_seed: int


def get_settings() -> Settings:
    """Return settings using environment overrides when provided."""
    raw_data_value = os.getenv("MARITIME_CBM_RAW_DATA_DIR")
    raw_data_dir = Path(raw_data_value).expanduser() if raw_data_value else DEFAULT_RAW_DATA_DIR
    if not raw_data_dir.is_absolute():
        raw_data_dir = PROJECT_ROOT / raw_data_dir

    seed_value = os.getenv("MARITIME_CBM_RANDOM_SEED", str(DEFAULT_RANDOM_SEED))
    try:
        random_seed = int(seed_value)
    except ValueError as error:
        raise ConfigurationError("MARITIME_CBM_RANDOM_SEED must be an integer") from error
    if random_seed < 0:
        raise ConfigurationError("MARITIME_CBM_RANDOM_SEED must be non-negative")

    return Settings(raw_data_dir=raw_data_dir, random_seed=random_seed)
