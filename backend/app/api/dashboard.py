"""Read-only dashboard data: GET /api/analytics and GET /api/model-performance.

Serves the JSON produced by ``python -m ml.dashboard_data`` (real dataset statistics and
model results). Re-read automatically when the files change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import get_settings

router = APIRouter(prefix="/api", tags=["dashboard"])

_cache: dict[Path, tuple[float, dict]] = {}


def dashboard_dir() -> Path:
    return get_settings().reports_dir / "dashboard"


def _load(path: Path) -> dict[str, Any]:
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=f"{path.name} not found. Run `python -m ml.dashboard_data` in backend/.")
    cached = _cache.get(path)
    if cached is None or cached[0] != mtime:
        _cache[path] = (mtime, json.loads(path.read_text(encoding="utf-8")))
    return _cache[path][1]


@router.get("/analytics", summary="Dataset statistics for the dashboard (UCI dataset 468)")
def analytics(directory: Path = Depends(dashboard_dir)) -> dict[str, Any]:
    return _load(directory / "analytics.json")


@router.get("/model-performance", summary="Model metrics, ROC/PR curves and calibration results")
def model_performance(directory: Path = Depends(dashboard_dir)) -> dict[str, Any]:
    return _load(directory / "model_performance.json")
