"""Application settings, loaded from environment variables or backend/.env."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "ShopSense AI API"
    environment: str = "development"

    # Origins allowed to call the API from a browser (local Next.js dev server).
    # Override with a JSON list, e.g. CORS_ORIGINS='["http://localhost:3000"]'
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    )
    # In development, also accept localhost on any port (Next.js falls back to
    # 3001, 3002, ... when 3000 is busy). Ignored outside development.
    cors_dev_origin_regex: str = r"^http://(localhost|127\.0\.0\.1)(:\d+)?$"

    # Interactive API docs (/docs, /redoc, /openapi.json). None = on in development only.
    enable_docs: bool | None = None
    # Max simultaneous SHAP explanations per process (CPU-heavy); extra requests get 429.
    explain_max_concurrency: int = Field(default=4, ge=1, le=64)
    # Build the SHAP explainer at startup (avoids a ~5-7 s first request). None = production only.
    preload_explainer: bool | None = None

    # Per-client requests per minute for the model and dashboard endpoints (/api/predict,
    # /api/explain, /api/analytics, ...). None = off in development, 120 otherwise; 0 = off.
    api_rate_limit_per_minute: int | None = Field(default=None, ge=0, le=100_000)

    # ---- Website Intelligence (POST /api/analyze-website) ----
    website_analysis_enabled: bool = True
    # Per-client sliding window (analyses per window) and global concurrent analyses.
    website_rate_limit: int = Field(default=10, ge=1, le=1000)
    website_rate_window_s: float = Field(default=600.0, ge=1)
    website_max_concurrency: int = Field(default=2, ge=1, le=32)
    # Optional Google PageSpeed Insights key. Without it, performance data is reported as unavailable.
    pagespeed_api_key: SecretStr | None = None

    data_dir: Path = BACKEND_DIR / "data"
    models_dir: Path = BACKEND_DIR / "models"
    reports_dir: Path = BACKEND_DIR / "reports"
    # Which saved model the API serves. None = the manifest's default
    # (without_page_values). Env: PREDICTION_MODEL_VARIANT=with_page_values
    prediction_model_variant: str | None = None

    @property
    def is_development(self) -> bool:
        return self.environment.lower() == "development"

    @property
    def explainer_preloaded(self) -> bool:
        return (not self.is_development) if self.preload_explainer is None else self.preload_explainer

    @property
    def api_rate_limit(self) -> int:
        """Effective per-minute limit; 0 means no limit."""
        if self.api_rate_limit_per_minute is None:
            return 0 if self.is_development else 120
        return self.api_rate_limit_per_minute

    @property
    def docs_enabled(self) -> bool:
        return self.is_development if self.enable_docs is None else self.enable_docs


@lru_cache
def get_settings() -> Settings:
    return Settings()
