from pathlib import Path

import pytest

from maritime_cbm.config import (
    DEFAULT_ALERT_ARTIFACT_DIR,
    DEFAULT_ALERT_REPORT_DIR,
    DEFAULT_DEPLOYMENT_CHECKPOINT_PATH,
    DEFAULT_DEPLOYMENT_CONTRACT_PATH,
    DEFAULT_M3_MODEL_ARTIFACT_DIR,
    DEFAULT_M3_MODEL_REPORT_DIR,
    DEFAULT_MODEL_ARTIFACT_DIR,
    DEFAULT_MODEL_REPORT_DIR,
    DEFAULT_RANDOM_SEED,
    DEFAULT_RAW_DATA_DIR,
    DEFAULT_SERVICE_REPORT_DIR,
    ConfigurationError,
    get_settings,
)


def test_get_settings_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MARITIME_CBM_RAW_DATA_DIR", raising=False)
    monkeypatch.delenv("MARITIME_CBM_MODEL_ARTIFACT_DIR", raising=False)
    monkeypatch.delenv("MARITIME_CBM_MODEL_REPORT_DIR", raising=False)
    monkeypatch.delenv("MARITIME_CBM_M3_MODEL_ARTIFACT_DIR", raising=False)
    monkeypatch.delenv("MARITIME_CBM_M3_MODEL_REPORT_DIR", raising=False)
    monkeypatch.delenv("MARITIME_CBM_ALERT_ARTIFACT_DIR", raising=False)
    monkeypatch.delenv("MARITIME_CBM_ALERT_REPORT_DIR", raising=False)
    monkeypatch.delenv("MARITIME_CBM_DEPLOYMENT_CONTRACT_PATH", raising=False)
    monkeypatch.delenv("MARITIME_CBM_DEPLOYMENT_CHECKPOINT_PATH", raising=False)
    monkeypatch.delenv("MARITIME_CBM_SERVICE_REPORT_DIR", raising=False)
    monkeypatch.delenv("MARITIME_CBM_RANDOM_SEED", raising=False)

    settings = get_settings()

    assert settings.raw_data_dir == DEFAULT_RAW_DATA_DIR
    assert settings.model_artifact_dir == DEFAULT_MODEL_ARTIFACT_DIR
    assert settings.model_report_dir == DEFAULT_MODEL_REPORT_DIR
    assert settings.m3_model_artifact_dir == DEFAULT_M3_MODEL_ARTIFACT_DIR
    assert settings.m3_model_report_dir == DEFAULT_M3_MODEL_REPORT_DIR
    assert settings.alert_artifact_dir == DEFAULT_ALERT_ARTIFACT_DIR
    assert settings.alert_report_dir == DEFAULT_ALERT_REPORT_DIR
    assert settings.deployment_contract_path == DEFAULT_DEPLOYMENT_CONTRACT_PATH
    assert settings.deployment_checkpoint_path == DEFAULT_DEPLOYMENT_CHECKPOINT_PATH
    assert settings.service_report_dir == DEFAULT_SERVICE_REPORT_DIR
    assert settings.random_seed == DEFAULT_RANDOM_SEED


def test_get_settings_accepts_environment_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MARITIME_CBM_RAW_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MARITIME_CBM_MODEL_ARTIFACT_DIR", "artifacts/test-modeling")
    monkeypatch.setenv("MARITIME_CBM_MODEL_REPORT_DIR", "reports/test-modeling")
    monkeypatch.delenv("MARITIME_CBM_M3_MODEL_ARTIFACT_DIR", raising=False)
    monkeypatch.delenv("MARITIME_CBM_M3_MODEL_REPORT_DIR", raising=False)
    monkeypatch.setenv("MARITIME_CBM_RANDOM_SEED", "7")
    monkeypatch.setenv("MARITIME_CBM_ALERT_ARTIFACT_DIR", "artifacts/test-alerting")
    monkeypatch.setenv("MARITIME_CBM_ALERT_REPORT_DIR", "reports/test-alerting")
    monkeypatch.setenv("MARITIME_CBM_DEPLOYMENT_CONTRACT_PATH", "config/test-deployment.json")
    monkeypatch.setenv(
        "MARITIME_CBM_DEPLOYMENT_CHECKPOINT_PATH",
        str(tmp_path / "deployment.pt"),
    )
    monkeypatch.setenv("MARITIME_CBM_SERVICE_REPORT_DIR", "reports/test-service")

    settings = get_settings()

    assert settings.raw_data_dir == tmp_path
    assert settings.model_artifact_dir == DEFAULT_MODEL_ARTIFACT_DIR.parent / "test-modeling"
    assert settings.model_report_dir == DEFAULT_MODEL_REPORT_DIR.parent / "test-modeling"
    assert settings.m3_model_artifact_dir == settings.model_artifact_dir / "m3"
    assert settings.m3_model_report_dir == settings.model_report_dir / "m3"
    assert settings.alert_artifact_dir == DEFAULT_ALERT_ARTIFACT_DIR.parent / "test-alerting"
    assert settings.alert_report_dir == DEFAULT_ALERT_REPORT_DIR.parent / "test-alerting"
    assert settings.deployment_contract_path == DEFAULT_DEPLOYMENT_CONTRACT_PATH.parent / (
        "test-deployment.json"
    )
    assert settings.deployment_checkpoint_path == tmp_path / "deployment.pt"
    assert settings.service_report_dir == DEFAULT_SERVICE_REPORT_DIR.parent / "test-service"
    assert settings.random_seed == 7


def test_get_settings_accepts_independent_m3_path_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MARITIME_CBM_MODEL_ARTIFACT_DIR", "artifacts/shared")
    monkeypatch.setenv("MARITIME_CBM_MODEL_REPORT_DIR", "reports/shared")
    monkeypatch.setenv("MARITIME_CBM_M3_MODEL_ARTIFACT_DIR", str(tmp_path / "m3-artifacts"))
    monkeypatch.setenv("MARITIME_CBM_M3_MODEL_REPORT_DIR", "reports/m3-explicit")

    settings = get_settings()

    assert settings.m3_model_artifact_dir == tmp_path / "m3-artifacts"
    assert settings.m3_model_report_dir == DEFAULT_MODEL_REPORT_DIR.parent / "m3-explicit"


@pytest.mark.parametrize("value", ["invalid", "-1"])
def test_get_settings_rejects_invalid_seed(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("MARITIME_CBM_RANDOM_SEED", value)

    with pytest.raises(ConfigurationError):
        get_settings()
