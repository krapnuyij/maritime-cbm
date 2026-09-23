from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from maritime_cbm.api.app import create_app
from maritime_cbm.api.deployment import (
    DEPLOYMENT_CANDIDATE_ID,
    DEPLOYMENT_MODEL_VERSION,
    DEPLOYMENT_POLICY_VERSION,
    DeploymentContract,
    DeploymentContractError,
    FeatureBounds,
    LoadedDeployment,
    torch_base_version,
)
from maritime_cbm.config import (
    DEFAULT_DEPLOYMENT_CHECKPOINT_PATH,
    DEFAULT_RAW_DATA_DIR,
    get_settings,
)
from maritime_cbm.data.features import select_model_features
from maritime_cbm.data.loader import load_raw_dataset
from maritime_cbm.data.schema import MODEL_FEATURE_COLUMNS, TARGET_COLUMNS
from maritime_cbm.modeling.preprocessing import SPEED_CENTERED_COLUMNS


@dataclass
class StubRegressor:
    """Predict deterministic ordered outputs without loading an official artifact."""

    def predict(self, features: object) -> np.ndarray:
        speeds = features["v"].to_numpy(dtype=np.float64)
        return np.column_stack((0.95 + speeds * 1e-4, 0.975 + speeds * 1e-4))


def _contract() -> DeploymentContract:
    return DeploymentContract(
        schema_version=1,
        model_version=DEPLOYMENT_MODEL_VERSION,
        candidate_id=DEPLOYMENT_CANDIDATE_ID,
        checkpoint_sha256="0" * 64,
        checkpoint_seed=42,
        checkpoint_scenario="state_group",
        torch_base_version=torch_base_version("2.14.0"),
        input_features=MODEL_FEATURE_COLUMNS,
        target_columns=TARGET_COLUMNS,
        allowed_speeds=(3.0, 6.0, 9.0),
        continuous_feature_bounds={
            feature: FeatureBounds(minimum=0.0, maximum=100.0) for feature in SPEED_CENTERED_COLUMNS
        },
        policy_version=DEPLOYMENT_POLICY_VERSION,
        watch_severity_threshold=0.5,
        alert_severity_threshold=0.8,
    )


def _deployment() -> LoadedDeployment:
    checkpoint = SimpleNamespace(regressor=StubRegressor())
    return LoadedDeployment(
        contract=_contract(),
        checkpoint=checkpoint,
        checkpoint_path=Path("synthetic-test-only.pt"),
    )


def _sensor_input(*, speed: float = 3.0) -> dict[str, float]:
    values = {feature: 50.0 for feature in MODEL_FEATURE_COLUMNS}
    values["v"] = speed
    return values


@pytest.fixture
def client() -> TestClient:
    with TestClient(create_app(deployment=_deployment())) as test_client:
        yield test_client


def test_health_and_model_info_report_loaded_contract(client: TestClient) -> None:
    health = client.get("/health")
    info = client.get("/model/info")

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": DEPLOYMENT_MODEL_VERSION,
    }
    assert info.status_code == 200
    assert info.json()["input_features"] == list(MODEL_FEATURE_COLUMNS)
    assert info.json()["allowed_speeds"] == [3.0, 6.0, 9.0]


def test_predict_and_batch_preserve_model_version_and_order(client: TestClient) -> None:
    single = client.post("/v1/condition/predict", json=_sensor_input(speed=6.0))
    batch = client.post(
        "/v1/condition/batch",
        json={"items": [_sensor_input(speed=9.0), _sensor_input(speed=3.0)]},
    )

    assert single.status_code == 200
    assert single.json()["model_version"] == DEPLOYMENT_MODEL_VERSION
    assert single.json()["prediction"] == pytest.approx({"kMc": 0.9506, "kMt": 0.9756})
    assert batch.status_code == 200
    assert [item["kMc"] for item in batch.json()["predictions"]] == pytest.approx([0.9509, 0.9503])


def test_predict_accepts_inclusive_bounds_and_rejects_unknown_speed(client: TestClient) -> None:
    lower = _sensor_input()
    lower["GTT"] = 0.0
    upper = _sensor_input()
    upper["GTT"] = 100.0

    assert client.post("/v1/condition/predict", json=lower).status_code == 200
    assert client.post("/v1/condition/predict", json=upper).status_code == 200

    response = client.post("/v1/condition/predict", json=_sensor_input(speed=4.0))
    assert response.status_code == 422
    assert response.json()["error"]["details"][0]["location"] == ["v"]


def test_predict_rejects_out_of_range_extra_missing_string_bool_and_nan(
    client: TestClient,
) -> None:
    invalid_payloads: list[tuple[dict[str, object], str]] = []
    outside = _sensor_input()
    outside["GTT"] = 100.1
    invalid_payloads.append((outside, "GTT"))
    extra: dict[str, object] = {**_sensor_input(), "extra": 1.0}
    invalid_payloads.append((extra, "extra"))
    missing: dict[str, object] = _sensor_input()
    del missing["mf"]
    invalid_payloads.append((missing, "mf"))
    string_value: dict[str, object] = _sensor_input()
    string_value["GTT"] = "50.0"
    invalid_payloads.append((string_value, "GTT"))
    boolean_value: dict[str, object] = _sensor_input()
    boolean_value["GTT"] = True
    invalid_payloads.append((boolean_value, "GTT"))

    for payload, field in invalid_payloads:
        response = client.post("/v1/condition/predict", json=payload)
        assert response.status_code == 422
        body = response.json()
        assert body["error"]["code"] == "VALIDATION_ERROR"
        assert field in body["error"]["details"][0]["location"]
        assert "input" not in body["error"]["details"][0]

    nan_body = (
        '{"v":3,"GTT":NaN,"GTn":50,"GGn":50,"Ts":50,"T48":50,'
        '"T2":50,"P48":50,"P2":50,"Pexh":50,"TIC":50,"mf":50}'
    )
    response = client.post(
        "/v1/condition/predict",
        content=nan_body,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert "input" not in response.json()["error"]["details"][0]


def test_batch_enforces_one_to_one_hundred_items(client: TestClient) -> None:
    empty = client.post("/v1/condition/batch", json={"items": []})
    too_large = client.post(
        "/v1/condition/batch",
        json={"items": [_sensor_input()] * 101},
    )

    assert empty.status_code == 422
    assert too_large.status_code == 422


def test_alert_evaluation_accepts_out_of_official_range_coefficients(client: TestClient) -> None:
    response = client.post("/v1/alert/evaluate", json={"kMc": 0.94, "kMt": 1.01})

    assert response.status_code == 200
    body = response.json()
    assert body["policy_version"] == DEPLOYMENT_POLICY_VERSION
    assert body["components"]["kMc"] == {"severity": pytest.approx(1.2), "state": "alert"}
    assert body["components"]["kMt"]["severity"] == pytest.approx(-0.4)
    assert body["overall"]["state"] == "alert"
    assert "실제 고장진단" in body["criteria"]["interpretation"]


def test_startup_fails_when_deployment_files_are_missing(tmp_path: Path) -> None:
    settings = replace(
        get_settings(),
        deployment_contract_path=tmp_path / "missing-contract.json",
        deployment_checkpoint_path=tmp_path / "missing-model.pt",
    )

    with pytest.raises(DeploymentContractError, match="contract file is missing"):
        with TestClient(create_app(settings=settings)):
            pass


@pytest.mark.integration
def test_official_checkpoint_serves_one_valid_release_row() -> None:
    data_path = DEFAULT_RAW_DATA_DIR / "data.txt"
    if not data_path.is_file() or not DEFAULT_DEPLOYMENT_CHECKPOINT_PATH.is_file():
        pytest.skip("official UCI release or deployment checkpoint is not available")
    frame = load_raw_dataset(data_path)
    payload = {
        key: float(value)
        for key, value in select_model_features(frame.iloc[[0]]).iloc[0].to_dict().items()
    }

    with TestClient(create_app()) as official_client:
        response = official_client.post("/v1/condition/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["model_version"] == DEPLOYMENT_MODEL_VERSION
    prediction = response.json()["prediction"]
    assert set(prediction) == set(TARGET_COLUMNS)
    assert np.isfinite(list(prediction.values())).all()
