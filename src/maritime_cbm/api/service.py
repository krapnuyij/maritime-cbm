"""Thread-safe inference and alert-policy application service."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import Lock

import numpy as np
import pandas as pd

from maritime_cbm.alerting.policy import AlertPolicyDecision, evaluate_alert_policy
from maritime_cbm.api.deployment import LoadedDeployment


@dataclass(frozen=True, slots=True)
class InputErrorDetail:
    """One domain-level input-contract violation."""

    location: tuple[str | int, ...]
    type: str
    message: str


class InputContractError(ValueError):
    """Raised when finite typed input violates the deployed model domain."""

    def __init__(self, detail: InputErrorDetail) -> None:
        super().__init__(detail.message)
        self.detail = detail


class InferenceService:
    """One loaded M3 model with serialized CPU inference and M4 policy access."""

    def __init__(self, deployment: LoadedDeployment) -> None:
        self.deployment = deployment
        self._inference_lock = Lock()

    @property
    def model_version(self) -> str:
        """Return the stable deployed model version."""
        return self.deployment.contract.model_version

    @property
    def policy_version(self) -> str:
        """Return the stable deployed policy version."""
        return self.deployment.contract.policy_version

    def _validate_record(
        self,
        record: Mapping[str, float],
        *,
        prefix: tuple[str | int, ...],
    ) -> dict[str, float]:
        contract = self.deployment.contract
        expected = set(contract.input_features)
        actual = set(record)
        if actual != expected:
            raise InputContractError(
                InputErrorDetail(
                    location=prefix,
                    type="feature_set",
                    message="입력 센서 이름이 배포 계약과 일치하지 않는다.",
                )
            )

        canonical = {feature: float(record[feature]) for feature in contract.input_features}
        speed = canonical["v"]
        if speed not in contract.allowed_speeds:
            raise InputContractError(
                InputErrorDetail(
                    location=(*prefix, "v"),
                    type="allowed_value",
                    message="v는 배포 계약에 기록된 9개 운항 속도 중 하나여야 한다.",
                )
            )
        for feature, bounds in contract.continuous_feature_bounds.items():
            value = canonical[feature]
            if not bounds.minimum <= value <= bounds.maximum:
                raise InputContractError(
                    InputErrorDetail(
                        location=(*prefix, feature),
                        type="inclusive_range",
                        message=(
                            f"{feature}는 상태 그룹 train 범위 "
                            f"[{bounds.minimum}, {bounds.maximum}] 안이어야 한다."
                        ),
                    )
                )
        return canonical

    def predict_many(self, records: Sequence[Mapping[str, float]]) -> np.ndarray:
        """Validate an ordered batch and return unclipped coefficient estimates."""
        if not records:
            raise InputContractError(
                InputErrorDetail(
                    location=("items",),
                    type="too_short",
                    message="배치에는 최소 한 건이 필요하다.",
                )
            )
        canonical = [
            self._validate_record(record, prefix=("items", index))
            for index, record in enumerate(records)
        ]
        frame = pd.DataFrame(canonical, columns=self.deployment.contract.input_features)
        with self._inference_lock:
            predictions = self.deployment.checkpoint.regressor.predict(frame)
        values = np.asarray(predictions, dtype=np.float64)
        expected_shape = (len(frame), len(self.deployment.contract.target_columns))
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise RuntimeError("Deployment model returned invalid predictions")
        return values

    def predict_one(self, record: Mapping[str, float]) -> np.ndarray:
        """Validate one feature record and return one unclipped prediction."""
        canonical = self._validate_record(record, prefix=())
        frame = pd.DataFrame([canonical], columns=self.deployment.contract.input_features)
        with self._inference_lock:
            predictions = self.deployment.checkpoint.regressor.predict(frame)
        values = np.asarray(predictions, dtype=np.float64)
        expected_shape = (1, len(self.deployment.contract.target_columns))
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise RuntimeError("Deployment model returned invalid predictions")
        return values[0]

    def evaluate_alert(self, k_mc: float, k_mt: float) -> AlertPolicyDecision:
        """Apply the existing M4 policy without clipping supplied coefficients."""
        contract = self.deployment.contract
        return evaluate_alert_policy(
            np.asarray([[k_mc, k_mt]], dtype=np.float64),
            watch_threshold=contract.watch_severity_threshold,
            alert_threshold=contract.alert_severity_threshold,
        )
