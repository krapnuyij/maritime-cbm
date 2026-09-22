"""FastAPI application factory for state estimation and alert evaluation."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from maritime_cbm.api.deployment import LoadedDeployment, load_deployment
from maritime_cbm.api.schemas import (
    AlertComponents,
    AlertCriteria,
    AlertEvaluationRequest,
    AlertEvaluationResponse,
    ConditionBatchRequest,
    ConditionBatchResponse,
    ConditionPrediction,
    ConditionPredictionResponse,
    ErrorEnvelope,
    HealthResponse,
    ModelInfoResponse,
    NumericBounds,
    SensorInput,
    SeverityState,
)
from maritime_cbm.api.service import InferenceService, InputContractError
from maritime_cbm.config import Settings, get_settings

VALIDATION_MESSAGE = "입력값이 API 계약을 만족하지 않는다."
POLICY_INTERPRETATION = (
    "시뮬레이터 열화 계수 기반 PoC 정책이며 실제 고장진단 또는 공식 경보 기준이 아니다."
)


def _validation_response(details: list[dict[str, object]]) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": VALIDATION_MESSAGE,
                "details": details,
            }
        },
    )


def _service(request: Request) -> InferenceService:
    service = getattr(request.app.state, "inference_service", None)
    if not isinstance(service, InferenceService):
        raise RuntimeError("Inference service is not initialized")
    return service


def create_app(
    *,
    settings: Settings | None = None,
    deployment: LoadedDeployment | None = None,
) -> FastAPI:
    """Create an app that loads and verifies one model during its lifespan."""
    resolved_settings = get_settings() if settings is None else settings

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        loaded = deployment
        if loaded is None:
            loaded = load_deployment(
                resolved_settings.deployment_contract_path,
                resolved_settings.deployment_checkpoint_path,
            )
        application.state.inference_service = InferenceService(loaded)
        yield
        del application.state.inference_service

    application = FastAPI(
        title="Maritime CBM API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        del request
        details = []
        for item in error.errors():
            location = list(item.get("loc", ()))
            if location and location[0] == "body":
                location = location[1:]
            details.append(
                {
                    "location": location,
                    "type": str(item.get("type", "validation_error")),
                    "message": str(item.get("msg", "invalid value")),
                }
            )
        return _validation_response(details)

    @application.exception_handler(InputContractError)
    async def input_contract_handler(
        request: Request,
        error: InputContractError,
    ) -> JSONResponse:
        del request
        detail = error.detail
        return _validation_response(
            [
                {
                    "location": list(detail.location),
                    "type": detail.type,
                    "message": detail.message,
                }
            ]
        )

    error_responses = {422: {"model": ErrorEnvelope}}

    @application.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        service = _service(request)
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_version=service.model_version,
        )

    @application.get("/model/info", response_model=ModelInfoResponse)
    async def model_info(request: Request) -> ModelInfoResponse:
        contract = _service(request).deployment.contract
        return ModelInfoResponse(
            model_version=contract.model_version,
            policy_version=contract.policy_version,
            input_features=list(contract.input_features),
            target_columns=list(contract.target_columns),
            allowed_speeds=list(contract.allowed_speeds),
            continuous_feature_bounds={
                feature: NumericBounds(minimum=bounds.minimum, maximum=bounds.maximum)
                for feature, bounds in contract.continuous_feature_bounds.items()
            },
            watch_severity_threshold=contract.watch_severity_threshold,
            alert_severity_threshold=contract.alert_severity_threshold,
        )

    @application.post(
        "/v1/condition/predict",
        response_model=ConditionPredictionResponse,
        responses=error_responses,
    )
    async def predict(request: Request, payload: SensorInput) -> ConditionPredictionResponse:
        service = _service(request)
        values = service.predict_one(payload.model_dump())
        return ConditionPredictionResponse(
            model_version=service.model_version,
            prediction=ConditionPrediction(kMc=float(values[0]), kMt=float(values[1])),
        )

    @application.post(
        "/v1/condition/batch",
        response_model=ConditionBatchResponse,
        responses=error_responses,
    )
    async def predict_batch(
        request: Request,
        payload: ConditionBatchRequest,
    ) -> ConditionBatchResponse:
        service = _service(request)
        values = service.predict_many([item.model_dump() for item in payload.items])
        return ConditionBatchResponse(
            model_version=service.model_version,
            predictions=[
                ConditionPrediction(kMc=float(row[0]), kMt=float(row[1])) for row in values
            ],
        )

    @application.post(
        "/v1/alert/evaluate",
        response_model=AlertEvaluationResponse,
        responses=error_responses,
    )
    async def evaluate_alert(
        request: Request,
        payload: AlertEvaluationRequest,
    ) -> AlertEvaluationResponse:
        service = _service(request)
        decision = service.evaluate_alert(payload.kMc, payload.kMt)
        contract = service.deployment.contract
        k_mc_severity = float(decision.severity[0, 0])
        k_mt_severity = float(decision.severity[0, 1])
        return AlertEvaluationResponse(
            policy_version=service.policy_version,
            components=AlertComponents(
                kMc=SeverityState(
                    severity=k_mc_severity,
                    state=str(decision.compressor_state[0]),
                ),
                kMt=SeverityState(
                    severity=k_mt_severity,
                    state=str(decision.turbine_state[0]),
                ),
            ),
            overall=SeverityState(
                severity=max(k_mc_severity, k_mt_severity),
                state=str(decision.overall_state[0]),
            ),
            criteria=AlertCriteria(
                watch_severity_threshold=contract.watch_severity_threshold,
                alert_severity_threshold=contract.alert_severity_threshold,
                interpretation=POLICY_INTERPRETATION,
            ),
        )

    return application


app = create_app()
