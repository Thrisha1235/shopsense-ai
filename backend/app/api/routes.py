"""Prediction endpoints: POST /api/predict and GET /api/model-info."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import get_settings
from app.schemas import ModelInfo, PredictionResponse, SessionFeatures
from app.services.prediction import (
    InputValidationError,
    ModelUnavailableError,
    PredictionService,
    load_service,
)

router = APIRouter(prefix="/api", tags=["prediction"])


log = logging.getLogger("shopsense.api")

# Public message only: the underlying reason can contain server file paths, so it is logged
# server-side instead of being returned to clients.
MODEL_UNAVAILABLE = "Prediction model unavailable. The server log has details."


def service_or_503(models_dir: Path, variant: str | None) -> PredictionService:
    try:
        return load_service(str(models_dir), variant)
    except ModelUnavailableError as exc:
        log.error("Prediction model unavailable: %s", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=MODEL_UNAVAILABLE) from exc


def get_service() -> PredictionService:
    s = get_settings()
    return service_or_503(s.models_dir, s.prediction_model_variant)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict whether a browsing session will end in a purchase",
    responses={422: {"description": "Invalid input"}, 503: {"description": "Model not available"}},
)
def predict(session: SessionFeatures, service: PredictionService = Depends(get_service)) -> PredictionResponse:
    try:
        result = service.predict(session.model_dump())
    except InputValidationError as exc:   # defensive: Pydantic should already have rejected it
        raise HTTPException(422, detail=str(exc)) from exc
    return PredictionResponse(**result)


@router.get(
    "/model-info",
    response_model=ModelInfo,
    summary="Model metadata, calibration status, input schema and limitations",
    responses={503: {"description": "Model not available"}},
)
def model_info(service: PredictionService = Depends(get_service)) -> ModelInfo:
    return ModelInfo(**service.info(SessionFeatures.model_json_schema()))
