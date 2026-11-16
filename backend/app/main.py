"""FastAPI entry point for ShopSense AI.

Run from the backend directory:
    uvicorn app.main:app --reload --port 8000              # development
    ENVIRONMENT=production uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import __version__
from app.api.dashboard import router as dashboard_router
from app.api.explain import router as explain_router
from app.api.routes import router as prediction_router
from app.api.website import router as website_router
from app.config import Settings, get_settings
from app.services.explanation import ExplanationUnavailableError, load_explanation_service
from app.services.prediction import ModelUnavailableError, load_service

log = logging.getLogger("shopsense")

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str
    timestamp: datetime


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Warm the model so the first request is fast. If it is missing, /health still
        # works and the prediction endpoints answer 503 until a model is trained.
        try:
            load_service(str(settings.models_dir), settings.prediction_model_variant)
        except ModelUnavailableError as exc:
            log.warning("Prediction model not loaded: %s", exc)
        if settings.explainer_preloaded:
            try:
                load_explanation_service(str(settings.models_dir), settings.prediction_model_variant)
            except ExplanationUnavailableError as exc:
                log.warning("SHAP explainer not preloaded: %s", exc)
        if not settings.is_development and any("localhost" in o or "127.0.0.1" in o for o in settings.cors_origins):
            log.warning("Non-development environment with localhost CORS origins; set CORS_ORIGINS.")
        yield

    docs = settings.docs_enabled
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="Purchasing-intention prediction and explainability API.",
        lifespan=lifespan,
        docs_url="/docs" if docs else None,
        redoc_url="/redoc" if docs else None,
        openapi_url="/openapi.json" if docs else None,
    )
    app.state.settings = settings

    # The browser frontend sends JSON with no cookies or auth, so credentials stay off
    # and only the methods/headers it actually uses are allowed.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=settings.cors_dev_origin_regex if settings.is_development else None,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
        max_age=600,
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        return response

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        """Liveness check used by the frontend and for local verification."""
        return HealthResponse(
            status="ok",
            service=settings.app_name,
            version=__version__,
            environment=settings.environment,
            timestamp=datetime.now(timezone.utc),
        )

    app.include_router(prediction_router)
    app.include_router(dashboard_router)
    # SHAP explainer loads lazily on the first /api/explain request (keeps startup fast).
    app.include_router(explain_router)
    # Website Intelligence: separate module, does not touch the prediction models.
    app.include_router(website_router)
    return app


app = create_app()
