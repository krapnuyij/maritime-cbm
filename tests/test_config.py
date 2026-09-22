from pathlib import Path

import pytest

from maritime_cbm.config import (
    DEFAULT_MODEL_ARTIFACT_DIR,
    DEFAULT_MODEL_REPORT_DIR,
    DEFAULT_RANDOM_SEED,
    DEFAULT_RAW_DATA_DIR,
    ConfigurationError,
    get_settings,
)


def test_get_settings_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MARITIME_CBM_RAW_DATA_DIR", raising=False)
    monkeypatch.delenv("MARITIME_CBM_MODEL_ARTIFACT_DIR", raising=False)
    monkeypatch.delenv("MARITIME_CBM_MODEL_REPORT_DIR", raising=False)
    monkeypatch.delenv("MARITIME_CBM_RANDOM_SEED", raising=False)

    settings = get_settings()

    assert settings.raw_data_dir == DEFAULT_RAW_DATA_DIR
    assert settings.model_artifact_dir == DEFAULT_MODEL_ARTIFACT_DIR
    assert settings.model_report_dir == DEFAULT_MODEL_REPORT_DIR
    assert settings.random_seed == DEFAULT_RANDOM_SEED


def test_get_settings_accepts_environment_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MARITIME_CBM_RAW_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MARITIME_CBM_MODEL_ARTIFACT_DIR", "artifacts/test-modeling")
    monkeypatch.setenv("MARITIME_CBM_MODEL_REPORT_DIR", "reports/test-modeling")
    monkeypatch.setenv("MARITIME_CBM_RANDOM_SEED", "7")

    settings = get_settings()

    assert settings.raw_data_dir == tmp_path
    assert settings.model_artifact_dir == DEFAULT_MODEL_ARTIFACT_DIR.parent / "test-modeling"
    assert settings.model_report_dir == DEFAULT_MODEL_REPORT_DIR.parent / "test-modeling"
    assert settings.random_seed == 7


@pytest.mark.parametrize("value", ["invalid", "-1"])
def test_get_settings_rejects_invalid_seed(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("MARITIME_CBM_RANDOM_SEED", value)

    with pytest.raises(ConfigurationError):
        get_settings()
