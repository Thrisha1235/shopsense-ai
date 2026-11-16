"""Explanation endpoints: POST /api/explain and GET /api/explain/global."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import get_settings
from app.schemas import ExplanationResponse, GlobalExplanationResponse, SessionFeatures
from app.services.explanation import (
    INTERPRETATION_NOTES,
    ExplanationService,
    ExplanationUnavailableError,
    load_explanation_service,
)
from app.services.prediction import InputValidationError

router = APIRouter(prefix="/api/explain", tags=["explanations"])
log = logging.getLogger("shopsense.api")

EXPLANATIONS_UNAVAILABLE = "Explanations unavailable. The server log has details."

# SHAP is CPU-heavy (~0.2 s per session): cap simultaneous explanations per process so a burst
# of requests cannot starve the prediction endpoints. Extra requests get 429 + Retry-After.
_explain_slots = threading.BoundedSemaphore(get_settings().explain_max_concurrency)


def explain_slot():
    if not _explain_slots.acquire(blocking=False):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="Too many explanation requests in progress. Retry shortly.",
                            headers={"Retry-After": "1"})
    try:
        yield
    finally:
        _explain_slots.release()


def explanation_service_or_503(models_dir: Path, variant: str | None) -> ExplanationService:
    try:
        return load_explanation_service(str(models_dir), variant)
    except ExplanationUnavailableError as exc:
        log.error("Explanations unavailable: %s", exc)   # may contain paths: log only
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=EXPLANATIONS_UNAVAILABLE) from exc


def get_explanation_service() -> ExplanationService:
    s = get_settings()
    return explanation_service_or_503(s.models_dir, s.prediction_model_variant)


def global_explanation_path() -> Path:
    return get_settings().reports_dir / "explanations" / "global_shap.json"


@router.post(
    "",
    response_model=ExplanationResponse,
    summary="Explain one session's raw model score with SHAP (plus the unchanged prediction)",
    responses={422: {"description": "Invalid input"}, 429: {"description": "Too many explanations in progress"},
               503: {"description": "Model or explainer unavailable"}},
    dependencies=[Depends(explain_slot)],
)
def explain(session: SessionFeatures,
            service: ExplanationService = Depends(get_explanation_service)) -> ExplanationResponse:
    try:
        result = service.explain(session.model_dump())
    except InputValidationError as exc:     # defensive: Pydantic should already have rejected it
        raise HTTPException(422, detail=str(exc)) from exc
    except RuntimeError as exc:              # consistency/additivity guard tripped: never return it
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Explanation failed a consistency check: {exc}") from exc
    return ExplanationResponse(**result)


@router.get(
    "/global",
    response_model=GlobalExplanationResponse,
    summary="Global feature importance (mean |SHAP| over held-out sessions)",
    responses={503: {"description": "Explanations or global importance unavailable"}},
)
def global_importance(service: ExplanationService = Depends(get_explanation_service),
                      path: Path = Depends(global_explanation_path)) -> GlobalExplanationResponse:
    if not path.exists():
        raise HTTPException(503, detail="Global importance not computed. Run `python -m ml.explain`.")
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("model_sha256") != service.record.get("model_sha256"):
        raise HTTPException(503, detail="Global importance was computed for a different model file. "
                                        "Re-run `python -m ml.explain`.")
    return GlobalExplanationResponse(
        variant=rec["variant"], method=rec["method"], explains=rec["explains"], output_units=rec["output_units"],
        baseline_score=rec["baseline_score"], baseline_description=rec["baseline_description"],
        typical_session_score=rec["typical_session_score"], n_sessions=rec["n_sessions"],
        sample_source=rec["global_sample"]["source"], max_additivity_error=rec["max_additivity_error"],
        mean_model_score=rec["mean_model_score"], aggregation=rec["aggregation"], features=rec["features"],
        interpretation_notes=INTERPRETATION_NOTES, created_at=rec.get("created_at"),
    )
