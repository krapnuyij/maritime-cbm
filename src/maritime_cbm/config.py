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
DEFAULT_M3_MODEL_ARTIFACT_DIR = DEFAULT_MODEL_ARTIFACT_DIR / "m3"
DEFAULT_M3_MODEL_REPORT_DIR = DEFAULT_MODEL_REPORT_DIR / "m3"
DEFAULT_ALERT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "alerting"
DEFAULT_ALERT_REPORT_DIR = PROJECT_ROOT / "reports" / "alerting"
DEFAULT_DEPLOYMENT_CONTRACT_PATH = PROJECT_ROOT / "config" / "deployment_model.json"
DEFAULT_DEPLOYMENT_CHECKPOINT_PATH = (
    DEFAULT_M3_MODEL_ARTIFACT_DIR / "checkpoints" / "state_group_seed_42.pt"
)
DEFAULT_SERVICE_REPORT_DIR = PROJECT_ROOT / "reports" / "service"


class ConfigurationError(ValueError):
    """Raised when a project setting is invalid."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings shared by data and modeling modules."""

    raw_data_dir: Path
    model_artifact_dir: Path
    model_report_dir: Path
    m3_model_artifact_dir: Path
    m3_model_report_dir: Path
    alert_artifact_dir: Path
    alert_report_dir: Path
    deployment_contract_path: Path
    deployment_checkpoint_path: Path
    service_report_dir: Path
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
    # M3-specific overrides take precedence. Without them, M3 follows the resolved
    # shared modeling root, including any parent-level environment override.
    m3_model_artifact_dir = _resolve_path(
        "MARITIME_CBM_M3_MODEL_ARTIFACT_DIR", model_artifact_dir / "m3"
    )
    m3_model_report_dir = _resolve_path("MARITIME_CBM_M3_MODEL_REPORT_DIR", model_report_dir / "m3")
    alert_artifact_dir = _resolve_path(
        "MARITIME_CBM_ALERT_ARTIFACT_DIR", DEFAULT_ALERT_ARTIFACT_DIR
    )
    alert_report_dir = _resolve_path("MARITIME_CBM_ALERT_REPORT_DIR", DEFAULT_ALERT_REPORT_DIR)
    deployment_contract_path = _resolve_path(
        "MARITIME_CBM_DEPLOYMENT_CONTRACT_PATH", DEFAULT_DEPLOYMENT_CONTRACT_PATH
    )
    deployment_checkpoint_path = _resolve_path(
        "MARITIME_CBM_DEPLOYMENT_CHECKPOINT_PATH", DEFAULT_DEPLOYMENT_CHECKPOINT_PATH
    )
    service_report_dir = _resolve_path(
        "MARITIME_CBM_SERVICE_REPORT_DIR", DEFAULT_SERVICE_REPORT_DIR
    )

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
        m3_model_artifact_dir=m3_model_artifact_dir,
        m3_model_report_dir=m3_model_report_dir,
        alert_artifact_dir=alert_artifact_dir,
        alert_report_dir=alert_report_dir,
        deployment_contract_path=deployment_contract_path,
        deployment_checkpoint_path=deployment_checkpoint_path,
        service_report_dir=service_report_dir,
        random_seed=random_seed,
    )
