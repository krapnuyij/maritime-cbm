"""Pydantic request and response schemas for the public API contract."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictRequestModel(BaseModel):
    """Reject coercion, non-finite numbers and undeclared request fields."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class SensorInput(StrictRequestModel):
    """Twelve named model features in the documented API order."""

    v: float
    GTT: float
    GTn: float
    GGn: float
    Ts: float
    T48: float
    T2: float
    P48: float
    P2: float
    Pexh: float
    TIC: float
    mf: float


class ConditionBatchRequest(StrictRequestModel):
    """Bounded state-estimation batch request."""

    items: list[SensorInput] = Field(min_length=1, max_length=100)


class AlertEvaluationRequest(StrictRequestModel):
    """Finite regression coefficients for separate policy evaluation."""

    kMc: float
    kMt: float


class ConditionPrediction(BaseModel):
    """Estimated simulator degradation-state coefficients."""

    kMc: float
    kMt: float


class ConditionPredictionResponse(BaseModel):
    """Single state-estimation response."""

    model_version: str
    prediction: ConditionPrediction


class ConditionBatchResponse(BaseModel):
    """Ordered batch state-estimation response."""

    model_version: str
    predictions: list[ConditionPrediction]


class SeverityState(BaseModel):
    """Unclipped degradation severity and its PoC policy state."""

    severity: float
    state: Literal["normal", "watch", "alert"]


class AlertComponents(BaseModel):
    """Component-level alert-policy outputs."""

    kMc: SeverityState
    kMt: SeverityState


class AlertCriteria(BaseModel):
    """Thresholds and interpretation boundary used for one policy response."""

    watch_severity_threshold: float
    alert_severity_threshold: float
    interpretation: str


class AlertEvaluationResponse(BaseModel):
    """Separate M4 policy response for supplied coefficient estimates."""

    policy_version: str
    components: AlertComponents
    overall: SeverityState
    criteria: AlertCriteria


class HealthResponse(BaseModel):
    """Process readiness after successful model loading."""

    status: Literal["ok"]
    model_loaded: bool
    model_version: str


class NumericBounds(BaseModel):
    """JSON response form for inclusive continuous-feature bounds."""

    minimum: float
    maximum: float


class ModelInfoResponse(BaseModel):
    """Versioned model, input and alert-policy contract metadata."""

    model_version: str
    policy_version: str
    input_features: list[str]
    target_columns: list[str]
    allowed_speeds: list[float]
    continuous_feature_bounds: dict[str, NumericBounds]
    watch_severity_threshold: float
    alert_severity_threshold: float


class ErrorDetail(BaseModel):
    """Sanitized validation detail without request values or internal paths."""

    location: list[str | int]
    type: str
    message: str


class ErrorBody(BaseModel):
    """Stable public error body."""

    code: str
    message: str
    details: list[ErrorDetail]


class ErrorEnvelope(BaseModel):
    """Stable public error envelope."""

    error: ErrorBody
